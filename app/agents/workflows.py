from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

from livekit import api, rtc
from livekit.protocol import room as proto_room

from livekit.agents import llm, stt, tts, utils, vad
from livekit.agents.job import get_job_context
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool
from livekit.agents.log import logger
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice import room_io
from livekit.agents.voice.agent import Agent, AgentTask
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.background_audio import (
    AudioConfig,
    AudioSource,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    PlayHandle,
)

if TYPE_CHECKING:
    from livekit.agents.voice.audio_recognition import TurnDetectionMode


BASE_INSTRUCTIONS = """
# Identity

You are an agent that is reaching out to a human agent for help. There has been a previous conversation
between you and a caller, the conversation history is included below.

# Goal

Your main goal is to give the human agent sufficient context about why the caller had called in,
so that the human agent could gain sufficient knowledge to help the caller directly.

# Context

In the conversation, user refers to the human agent, caller refers to the person who's transcript is included.
Remember, you are not speaking to the caller right now, you are speaking to the human agent.

Once the human agent has confirmed, you should call the tool `connect_to_caller` to connect them to the caller.

Start by giving them a summary of the conversation so far, and answer any questions they might have.

## Conversation history with caller
{conversation_history}
## End of conversation history with caller

You are talking to the human agent now,
give a brief introduction of the conversation so far, and ask if they want to connect to the caller.
"""
@dataclass
class WarmTransferResult:
    human_agent_identity: str


class WarmTransferTask(AgentTask[WarmTransferResult]):
    def __init__(
        self,
        target_phone_number: str,
        *,
        hold_audio: NotGivenOr[AudioSource | AudioConfig | list[AudioConfig] | None] = NOT_GIVEN,
        sip_trunk_id: NotGivenOr[str] = NOT_GIVEN,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=self.get_instructions(
                chat_ctx=chat_ctx, extra_instructions=extra_instructions
            ),
            chat_ctx=NOT_GIVEN,  # don't pass the chat_ctx
            turn_detection=turn_detection,
            tools=tools or [],
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

        self._caller_room: rtc.Room | None = None
        self._human_agent_sess: AgentSession | None = None
        self._human_agent_failed_fut: asyncio.Future[None] = asyncio.Future()
        self._human_agent_identity = "human-agent-sip"

        self._target_phone_number = target_phone_number
        self._sip_trunk_id = (
            sip_trunk_id if is_given(sip_trunk_id) else os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "")
        )
        if not self._sip_trunk_id:
            raise ValueError(
                "`LIVEKIT_SIP_OUTBOUND_TRUNK` environment variable or `sip_trunk_id` argument must be set"
            )

        # background audio and io
        self._background_audio = BackgroundAudioPlayer()
        self._hold_audio_handle: PlayHandle | None = None
        self._hold_audio = (
            cast(Optional[Union[AudioSource, AudioConfig, list[AudioConfig]]], hold_audio)
            if is_given(hold_audio)
            else AudioConfig(BuiltinAudioClip.HOLD_MUSIC, volume=0.8)
        )

        self._original_io_state: dict[str, bool] = {}

    def get_instructions(
        self, *, chat_ctx: NotGivenOr[llm.ChatContext], extra_instructions: str = ""
    ) -> str:
        # users can override this method if they want to customize the entire instructions
        prev_convo = ""
        if chat_ctx:
            context_copy = chat_ctx.copy(
                exclude_empty_message=True, exclude_instructions=True, exclude_function_call=True
            )
            for msg in context_copy.items:
                if msg.type != "message":
                    continue
                role = "Caller" if msg.role == "user" else "Assistant"
                prev_convo += f"{role}: {msg.text_content}\n"
        return BASE_INSTRUCTIONS.format(conversation_history=prev_convo) + extra_instructions

    async def on_enter(self) -> None:
        logger.info(f"WarmTransferTask: on_enter calling {self._target_phone_number}")

        # 1. Start Hold Music for the Caller
        if self._hold_audio is not None:
            await self._background_audio.start(room=self.session.room, agent_session=self.session)
            self._hold_audio_handle = self._background_audio.play(self._hold_audio, loop=True)

        try:
            # Dial the human supervisor
            dial_human_agent_task = asyncio.create_task(self._dial_human_agent())
            
            # 2. Selective Isolation for Privacy & Hold Music
            # Use LiveKit API to manage track-level subscriptions
            api_url = os.getenv("LIVEKIT_URL", "").replace("wss://", "https://")
            lkapi = api.LiveKitAPI(
                url=api_url,
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET")
            )

            # Find Caller
            caller = next((p for p in self.session.room.remote_participants.values() if p.identity != self._human_agent_identity), None)
            if caller:
                # Isolate Caller from AI Speech (so they only hear background audio)
                speech_track_sid = None
                for pub in self.session.room.local_participant.tracks.values():
                    if pub.name != "background_audio" and pub.kind == rtc.TrackKind.KIND_AUDIO:
                        speech_track_sid = pub.sid
                        break
                
                if speech_track_sid:
                    await lkapi.room.update_subscriptions(
                        proto_room.UpdateSubscriptionsRequest(
                            room=self.session.room.name,
                            identity=caller.identity,
                            track_sids=[speech_track_sid],
                            subscribe=False
                        )
                    )
                
                # Unsubscribe AI Agent from Caller (to prevent accidental interruptions during summary)
                caller_track_sid = next((t.sid for t in caller.tracks.values() if t.kind == rtc.TrackKind.KIND_AUDIO), None)
                if caller_track_sid:
                    await lkapi.room.update_subscriptions(
                        proto_room.UpdateSubscriptionsRequest(
                            room=self.session.room.name,
                            identity=self.session.room.local_participant.identity,
                            track_sids=[caller_track_sid],
                            subscribe=False
                        )
                    )
            
            await lkapi.aclose()

            # Wait for human to answer
            done, _ = await asyncio.wait(
                (dial_human_agent_task, self._human_agent_failed_fut),
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            if dial_human_agent_task in done:
                logger.info(f"Human agent answered: {self._human_agent_identity}")
                
                # 3. Switch AI Focus to Human Agent (Ensures human's mic is heard)
                if self._human_agent_identity:
                    self.session.set_participant(self._human_agent_identity)

                # Stop hold music so it doesn't bleed into the consultation
                if self._background_audio:
                    await self._background_audio.aclose()

                # 4. Summarize to Human Supervisor
                # We use the task's instructions which contain the conversation history summary
                await self.session.generate_reply(instructions=self.instructions)
            else:
                raise RuntimeError("Dialing failed or timed out")

        except Exception:
            logger.exception("Could not transfer to human agent")
            self._set_result(ToolError("Transfer failed: could not reach supervisor."))
        finally:
            if not dial_human_agent_task.done():
                await utils.aio.cancel_and_wait(dial_human_agent_task)
            
    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def connect_to_caller(self, reason: str = "") -> None:
        """Merges the call by connecting the caller to the human supervisor."""
        logger.info("Merging call: Connecting caller to human agent.")
        
        # We don't need to restore permissions here if we didn't use update_participant.
        # Track subscriptions will be cleaned up when AI leaves.
        # But for completeness, we can focus back on caller or just let it be.
        
        await self.session.generate_reply(
            instructions="Tell the human agent and the caller that you are connecting them now and then leave."
        )
        await self.session.wait_for_playout()
        self.complete(WarmTransferResult(human_agent_identity=self._human_agent_identity))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_transfer(self, reason: str) -> None:
        """Handles the case when the human agent explicitly declines to connect to the caller.

        Args:
            reason: A short explanation of why the human agent declined to connect to the caller
        """
        self._set_result(ToolError(f"human agent declined to connect: {reason}"))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def voicemail_detected(self, reason: str = "") -> None:
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        self._set_result(ToolError("voicemail detected"))

    def _on_human_agent_room_close(self, reason: rtc.DisconnectReason.ValueType) -> None:
        logger.debug(
            "human agent's room closed",
            extra={"reason": rtc.DisconnectReason.Name(reason)},
        )
        with contextlib.suppress(asyncio.InvalidStateError):
            self._human_agent_failed_fut.set_result(None)

        self._set_result(ToolError(f"room closed: {rtc.DisconnectReason.Name(reason)}"))

    def _on_caller_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.kind not in (
            rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
            rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        ):
            return

        logger.info(f"participant disconnected from caller room: {participant.identity}, closing")

        assert self._caller_room is not None
        self._caller_room.off("participant_disconnected", self._on_caller_participant_disconnected)
        job_ctx = get_job_context()
        job_ctx.delete_room(room_name=self._caller_room.name)

    def _set_result(self, result: WarmTransferResult | Exception) -> None:
        if self.done():
            return

        # Do NOT shutdown human_agent_sess here if you want a smooth handoff.
        # Just stop the audio.
        if self._hold_audio_handle:
            self._hold_audio_handle.stop()
            self._hold_audio_handle = None

        self._set_io_enabled(True)
        self.complete(result)

    async def _dial_human_agent(self) -> None:
        assert self._caller_room is not None

        job_ctx = get_job_context()

        logger.debug(
            "dialing human agent into caller room",
            extra={"human_agent_identity": self._human_agent_identity},
        )

        # dial the human agent directly into the caller's room
        await job_ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=self._sip_trunk_id,
                sip_call_to=self._target_phone_number,
                room_name=self._caller_room.name,
                participant_identity=self._human_agent_identity,
                wait_until_answered=True,
            )
        )

    async def _merge_calls(self) -> None:
        assert self._caller_room is not None
        # We don't need to assert human_agent_sess here if we are just opening audio
        
        logger.info("Merging calls: stopping hold music")

        if self._hold_audio_handle:
            self._hold_audio_handle.stop()
            self._hold_audio_handle = None
        
        # Open the IO for the original agent
        self._set_io_enabled(True)

        # Instead of shutting down, we let the Task result handle the cleanup
        logger.debug("Calls successfully merged.")

    def _set_io_enabled(self, enabled: bool) -> None:
        input = self.session.input
        output = self.session.output

        if not self._original_io_state:
            self._original_io_state = {
                "audio_input": input.audio_enabled,
                "video_input": input.video_enabled,
                "audio_output": output.audio_enabled,
                "transcription_output": output.transcription_enabled,
                "video_output": output.video_enabled,
            }

        if input.audio:
            input.set_audio_enabled(enabled and self._original_io_state["audio_input"])
        if input.video:
            input.set_video_enabled(enabled and self._original_io_state["video_input"])
        if output.audio:
            output.set_audio_enabled(enabled and self._original_io_state["audio_output"])
        if output.transcription:
            output.set_transcription_enabled(
                enabled and self._original_io_state["transcription_output"]
            )
        if output.video:
            output.set_video_enabled(enabled and self._original_io_state["video_output"])

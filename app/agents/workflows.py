from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

from livekit import api, rtc

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
        job_ctx = get_job_context()
        self._caller_room = job_ctx.room
        
        # 1. Start Hold Music for the Caller
        if self._hold_audio is not None:
            await self._background_audio.start(room=self._caller_room)
            self._hold_audio_handle = self._background_audio.play(self._hold_audio, loop=True)

        # 2. Isolate the caller from hearing the human agent or being heard
        # Find the caller's identity. The caller is typically a SIP or STANDARD participant in the room.
        # There should only be one remote participant if this is a 1:1 call.
        caller_identity = None
        for participant in self._caller_room.remote_participants.values():
            if participant.kind in (
                rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
                rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
            ):
                caller_identity = participant.identity
                break
        
        if caller_identity:
            try:
                from livekit.protocol import models, room as room_proto
                await job_ctx.api.room.update_participant(
                    room_proto.UpdateParticipantRequest(
                        room=self._caller_room.name,
                        identity=caller_identity,
                        permission=models.ParticipantPermission(
                            can_subscribe=True, # MUST be True for DTMF/Hold Music!
                            can_publish=False,  # Still quiet
                        )
                    )
                )
                logger.debug(f"Limited caller {caller_identity} to listening only")
            except Exception as e:
                logger.error(f"Failed to limit caller: {e}")
        else:
            logger.warning("Could not find caller identity to limit.")

        try:
            # 3. Start the transfer process (Dial human supervisor into this room)
            # wait_until_answered is True, so this task completes when supervisor answers.
            dial_human_agent_task = asyncio.create_task(self._dial_human_agent())
            
            done, _ = await asyncio.wait(
                (dial_human_agent_task, self._human_agent_failed_fut),
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            if dial_human_agent_task not in done:
                raise RuntimeError("Dialing failed or timed out")
            
            # Identify the supervisor participant who just joined
            supervisor_identity = self._human_agent_identity
            
            # Switch AI focus to the supervisor so its "ears" (STT) hear the supervisor
            if hasattr(self.session, "_room_io") and self.session._room_io:
                self.session._room_io.set_participant(supervisor_identity)
                logger.debug(f"Switched AI STT focus to supervisor {supervisor_identity}")

            # Isolate caller and supervisor
            if caller_identity:
                try:
                    from livekit.protocol import room as room_proto
                    
                    # 1. Isolate CALLER from AI and Supervisor
                    # Caller should only hear hold music
                    caller_unsubs = []
                    
                    # AI's tracks (unsubs from Microphone, but NOT background audio)
                    for track_pub in self._caller_room.local_participant.track_publications.values():
                        if track_pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
                            caller_unsubs.append(track_pub.sid)
                            
                    # Supervisor voice track
                    supervisor = self._caller_room.remote_participants.get(supervisor_identity)
                    if supervisor:
                        # Wait a moment for supervisor to actually publish tracks
                        for _ in range(10): # Max 1s wait
                            if any(t.kind == rtc.TrackKind.KIND_AUDIO for t in supervisor.track_publications.values()):
                                break
                            await asyncio.sleep(0.1)

                        for track_pub in supervisor.track_publications.values():
                            if track_pub.kind == rtc.TrackKind.KIND_AUDIO:
                                caller_unsubs.append(track_pub.sid)

                    if caller_unsubs:
                        await job_ctx.api.room.update_subscriptions(
                            room_proto.UpdateSubscriptionsRequest(
                                room=self._caller_room.name,
                                identity=caller_identity,
                                track_sids=caller_unsubs,
                                subscribe=False
                            )
                        )
                    
                    # 2. Isolate SUPERVISOR from Hold Music
                    # Supervisor should hear AI and Caller
                    # Supervisor should NOT hear Hold Music (Background Audio)
                    supervisor_unsubs = []
                    for track_pub in self._caller_room.local_participant.track_publications.values():
                        # If it's NOT the primary microphone track, it's hold music
                        if track_pub.source != rtc.TrackSource.SOURCE_MICROPHONE:
                            supervisor_unsubs.append(track_pub.sid)
                            
                    if supervisor_unsubs:
                        await job_ctx.api.room.update_subscriptions(
                            room_proto.UpdateSubscriptionsRequest(
                                room=self._caller_room.name,
                                identity=supervisor_identity,
                                track_sids=supervisor_unsubs,
                                subscribe=False
                            )
                        )
                        
                    logger.debug(f"Isolation complete: Caller unsubs={caller_unsubs}, Supervisor unsubs={supervisor_unsubs}")
                except Exception as e:
                    logger.error(f"Failed to perform cross-isolation: {e}")

            # Stop the hold music as soon as human answers? 
            # Actually, user wants caller to hear music/dtmf WHILE the agent summarizes.
            # So we keep hold music playing.
            
            # Introduce the human supervisor
            # Note: Because we unsubscribed the caller from AI voice, they will NOT hear this.
            self.session.generate_reply(
                instructions="The human agent has just joined the call. Please give a brief introduction of the conversation so far, and ask if they want to connect to the caller.",
            )

        except Exception:
            logger.exception("Could not dial human agent")
            self._set_result(ToolError("Transfer failed: could not reach supervisor."))
        finally:
            if not dial_human_agent_task.done():
                await utils.aio.cancel_and_wait(dial_human_agent_task)
            
    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def connect_to_caller(self, reason: str = "") -> None:
        """Called when the human agent wants to connect to the caller."""
        logger.debug("connecting to caller")
        assert self._caller_room is not None

        # Restore caller permissions so they can hear and be heard again
        job_ctx = get_job_context()
        caller_identity = None
        for participant in self._caller_room.remote_participants.values():
            if participant.kind in (
                rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
                rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
            ) and participant.identity != self._human_agent_identity:
                caller_identity = participant.identity
                break
                
        if caller_identity:
            try:
                from livekit.protocol import models, room as room_proto
                
                # Switch STT focus back to the caller (standard behavior)
                if hasattr(self.session, "_room_io") and self.session._room_io:
                    self.session._room_io.set_participant(caller_identity)

                # 1. Restore permissions
                await job_ctx.api.room.update_participant(
                    room_proto.UpdateParticipantRequest(
                        room=self._caller_room.name,
                        identity=caller_identity,
                        permission=models.ParticipantPermission(
                            can_subscribe=True, 
                            can_publish=True,
                        )
                    )
                )
                
                # 2. Gather ALL audio track SIDs to ensure explicit resubscription
                all_audio_sids = []
                supervisor = self._caller_room.remote_participants.get(self._human_agent_identity)
                caller = self._caller_room.remote_participants.get(caller_identity)
                
                # Check Local Participant (AI Voice + Background/Hold Music)
                for tp in self._caller_room.local_participant.track_publications.values():
                    if tp.kind == rtc.TrackKind.KIND_AUDIO:
                        all_audio_sids.append(tp.sid)
                
                if supervisor:
                    for tp in supervisor.track_publications.values():
                        if tp.kind == rtc.TrackKind.KIND_AUDIO:
                            all_audio_sids.append(tp.sid)
                
                if caller:
                    for tp in caller.track_publications.values():
                        if tp.kind == rtc.TrackKind.KIND_AUDIO:
                            all_audio_sids.append(tp.sid)

                # 3. Explicitly resubscribe CALLER to all audio
                await job_ctx.api.room.update_subscriptions(
                    room_proto.UpdateSubscriptionsRequest(
                        room=self._caller_room.name,
                        identity=caller_identity,
                        track_sids=all_audio_sids,
                        subscribe=True, 
                    )
                )

                # 4. Explicitly resubscribe SUPERVISOR to all audio
                await job_ctx.api.room.update_subscriptions(
                    room_proto.UpdateSubscriptionsRequest(
                        room=self._caller_room.name,
                        identity=self._human_agent_identity,
                        track_sids=all_audio_sids,
                        subscribe=True, 
                    )
                )
                logger.debug(f"Restored full permissions and explicit subscriptions for both humans.")
            except Exception as e:
                logger.error(f"Failed to restore permissions for caller: {e}")

        await self._merge_calls()
        self._set_result(WarmTransferResult(human_agent_identity=self._human_agent_identity))

        # when the caller or human agent leaves the room, we'll delete the room
        self._caller_room.on("participant_disconnected", self._on_caller_participant_disconnected)

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

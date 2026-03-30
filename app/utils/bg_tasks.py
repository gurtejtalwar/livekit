import asyncio
import logging

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    def __init__(self):
        self.tasks: set[asyncio.Task] = set()

    def create(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)

        task.add_done_callback(self._handle_done)
        return task

    def _handle_done(self, task: asyncio.Task):
        self.tasks.discard(task)

        try:
            task.result()
        except Exception as e:
            logger.exception("Background task failed", exc_info=e)

    async def wait_all(self):
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

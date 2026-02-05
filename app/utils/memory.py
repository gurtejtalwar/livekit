import psutil

def memory_pressure():
    vm = psutil.virtual_memory()
    return vm.percent > 75

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss  # in bytes

async def memory_watcher(kb_manager: KBManager):
    while True:
        await asyncio.sleep(5)

        if not memory_pressure():
            continue

        now = time.time()

        # unload oldest, unused KBs first
        candidates = [
            (agent_id, kb)
            for agent_id, kb in kb_manager._kbs.items()
            if ACTIVE_CALLS.get(agent_id, 0) == 0
        ]

        # sort by last_used (LRU)
        candidates.sort(key=lambda x: x[1].last_used)

        for agent_id, _ in candidates:
            await kb_manager.unload_kb(agent_id)

            if not memory_pressure():
                break

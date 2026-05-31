import threading
import time
import unittest


class MPSSerializationTests(unittest.TestCase):
    def test_serialized_mps_execution_allows_only_one_thread_at_a_time(self):
        from lada.utils.mps_utils import serialized_mps_execution

        active_count = 0
        max_active_count = 0
        state_lock = threading.Lock()
        first_entered = threading.Event()

        def worker(index):
            nonlocal active_count, max_active_count
            with serialized_mps_execution():
                with state_lock:
                    active_count += 1
                    max_active_count = max(max_active_count, active_count)
                    if index == 0:
                        first_entered.set()
                time.sleep(0.05)
                with state_lock:
                    active_count -= 1

        first = threading.Thread(target=worker, args=(0,))
        second = threading.Thread(target=worker, args=(1,))
        first.start()
        self.assertTrue(first_entered.wait(timeout=1))
        second.start()
        first.join()
        second.join()

        self.assertEqual(max_active_count, 1)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

import process_video_parallel as pvp


class ProcessVideoParallelExecutorSelectionTests(unittest.TestCase):
    def test_argparser_accepts_thread_executor(self):
        parser = pvp.build_arg_parser()
        args = parser.parse_args([
            "--input", "in.mp4",
            "--output", "out.mp4",
            "--executor", "thread",
        ])

        self.assertEqual(args.executor, "thread")

    def test_create_parallel_executor_uses_thread_pool_when_requested(self):
        args = mock.Mock(parallel_workers=2, executor="thread")

        with mock.patch.object(pvp, "ThreadPoolExecutor") as thread_pool:
            executor = pvp.create_parallel_executor(args)

        self.assertEqual(executor, thread_pool.return_value)
        thread_pool.assert_called_once_with(max_workers=2)

    def test_create_parallel_executor_uses_process_pool_by_default(self):
        args = mock.Mock(parallel_workers=3, executor="process")

        with mock.patch.object(pvp, "ProcessPoolExecutor") as process_pool:
            executor = pvp.create_parallel_executor(args)

        self.assertEqual(executor, process_pool.return_value)
        self.assertEqual(process_pool.call_args.kwargs["max_workers"], 3)
        self.assertEqual(process_pool.call_args.kwargs["max_tasks_per_child"], 1)


if __name__ == "__main__":
    unittest.main()

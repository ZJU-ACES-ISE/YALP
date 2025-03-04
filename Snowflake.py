# -*- coding: utf-8 -*-
import time


class Snowflake:
    """
    使用雪花算法生成唯一ID
    需要保证worker_id和datacenter_id在使用时是唯一的
    """
    def __init__(self, worker_id, datacenter_id):
        self.worker_id = worker_id
        self.datacenter_id = datacenter_id
        self.sequence = 0
        self.last_timestamp = -1

    def generate_id(self):
        timestamp = self.current_timestamp()

        if timestamp < self.last_timestamp:
            raise Exception("Invalid system clock!")

        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & 4095
            if self.sequence == 0:
                timestamp = self.wait_next_millis(self.last_timestamp)
        else:
            self.sequence = 0

        self.last_timestamp = timestamp

        return (
                (timestamp << 22)
                | (self.datacenter_id << 17)
                | (self.worker_id << 12)
                | self.sequence
        )

    def wait_next_millis(self, last_timestamp):
        timestamp = self.current_timestamp()
        while timestamp <= last_timestamp:
            timestamp = self.current_timestamp()
        return timestamp

    def current_timestamp(self):
        return int(time.time() * 1000)


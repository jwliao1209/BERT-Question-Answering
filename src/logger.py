from src.tracker import MetricTracker


class Logger:
    def __inti__(self, writer):
        self.progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.metric_tracker = MetricTracker()
        self.writer = writer
        
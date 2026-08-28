"""Concise terminal progress reporting for model training."""

from __future__ import annotations

import sys
from typing import Optional, TextIO

from tqdm.auto import tqdm


class TrainingProgress:
    """Display setup information and one ETA-bearing optimizer-step bar."""

    def __init__(self, *, enabled: bool = True, file: Optional[TextIO] = None) -> None:
        self.enabled = enabled
        self.file = file
        self.epochs = 0
        self._bar: Optional[tqdm] = None

    def message(self, value: str) -> None:
        if not self.enabled:
            return
        output = self.file or sys.stderr
        if self._bar is None:
            print(value, file=output, flush=True)
        else:
            tqdm.write(value, file=output)

    def start(
        self,
        *,
        run: str,
        device: str,
        samples: int,
        epochs: int,
        batch_size: int,
        batches_per_epoch: int,
        steps_per_epoch: int,
        total_steps: int,
        completed_steps: int,
        gradient_accumulation_steps: int,
        amp: bool,
        trainable_parameters: int,
        total_parameters: int,
        encoder_learning_rate: float,
        head_learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        learning_rate_schedule: str,
        learning_rate_schedule_steps: int,
    ) -> None:
        if not self.enabled:
            return
        self.epochs = epochs
        output = self.file or sys.stderr
        print(
            "Training setup: "
            f"run={run}, device={device}, samples={samples}, epochs={epochs}, "
            f"batch_size={batch_size}, batches/epoch={batches_per_epoch}, "
            f"optimizer_steps/epoch={steps_per_epoch}, total_steps={total_steps}, "
            f"gradient_accumulation={gradient_accumulation_steps}, "
            f"AMP={'on' if amp else 'off'}",
            file=output,
            flush=True,
        )
        percentage = (
            100.0 * trainable_parameters / total_parameters if total_parameters else 0.0
        )
        print(
            "Model parameters: "
            f"{trainable_parameters:,} trainable / {total_parameters:,} total "
            f"({percentage:.1f}% trainable)",
            file=output,
            flush=True,
        )
        print(
            "Optimization: AdamW, "
            f"encoder_lr={encoder_learning_rate:.2e}, "
            f"decode_head_lr={head_learning_rate:.2e}, "
            f"weight_decay={weight_decay:g}, warmup_steps={warmup_steps}, "
            f"schedule={learning_rate_schedule}/{learning_rate_schedule_steps} steps",
            file=output,
            flush=True,
        )
        self._bar = tqdm(
            total=total_steps,
            initial=min(completed_steps, total_steps),
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            file=self.file,
        )

    def step(
        self,
        *,
        epoch: int,
        loss: float,
        learning_rate: float,
        samples_per_second: float,
    ) -> None:
        if self._bar is None:
            return
        self._bar.update(1)
        self._bar.set_postfix(
            {
                "epoch": f"{epoch + 1}/{self.epochs}",
                "loss": f"{loss:.4f}",
                "lr": f"{learning_rate:.2e}",
                "samples/s": f"{samples_per_second:.2f}",
            },
            refresh=True,
        )

    def phase(self, *, epoch: int, name: str) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(
                f"epoch={epoch + 1}/{self.epochs}, phase={name}",
                refresh=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

"""Terminal progress reporting for long dataset-generation runs."""

from __future__ import annotations

from types import TracebackType
from typing import Optional, TextIO, Type

from tqdm.auto import tqdm


class DatasetGenerationProgress:
    """Two-level scene/view progress display backed by tqdm."""

    def __init__(self, *, enabled: bool = True, file: Optional[TextIO] = None) -> None:
        self.enabled = enabled
        self.file = file
        self.total_samples = 0
        self.stored_samples = 0
        self.accepted = 0
        self.rejected = 0
        self.existing = 0
        self._scenes: Optional[tqdm] = None
        self._views: Optional[tqdm] = None

    def __enter__(self) -> "DatasetGenerationProgress":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def start(self, total_scenes: int, total_samples: int, existing_samples: int) -> None:
        self.total_samples = total_samples
        self.stored_samples = existing_samples
        self._scenes = tqdm(
            total=total_scenes,
            desc="Scenes",
            unit="scene",
            dynamic_ncols=True,
            disable=not self.enabled,
            file=self.file,
        )
        self._set_scene_postfix()

    def scene_start(self, scene_index: int, scene_id: str) -> None:
        self._close_views()
        self.accepted = 0
        self.rejected = 0
        self.existing = 0
        if self._scenes is not None:
            total = self._scenes.total or 0
            self._scenes.set_postfix_str(
                f"current={scene_index}/{total} {scene_id}, "
                f"stored={self.stored_samples}/{self.total_samples}",
                refresh=True,
            )

    def samples_start(self, scene_id: str, total_samples: int) -> None:
        self._views = tqdm(
            total=total_samples,
            desc=f"  {scene_id}",
            unit="view",
            position=1,
            leave=False,
            dynamic_ncols=True,
            disable=not self.enabled,
            file=self.file,
        )
        self._set_view_postfix()

    def sample_complete(self, status: str) -> None:
        if status == "accepted":
            self.accepted += 1
            self.stored_samples += 1
        elif status == "rejected":
            self.rejected += 1
        elif status == "existing":
            self.existing += 1
        else:
            raise ValueError(f"Unknown generation progress status: {status}")
        if self._views is not None:
            self._set_view_postfix(refresh=False)
            self._views.update(1)

    def scene_complete(self) -> None:
        self._close_views()
        if self._scenes is not None:
            self._scenes.update(1)
            self._set_scene_postfix()

    def close(self) -> None:
        self._close_views()
        if self._scenes is not None:
            self._scenes.close()
            self._scenes = None

    def _set_scene_postfix(self) -> None:
        if self._scenes is not None:
            self._scenes.set_postfix_str(
                f"stored={self.stored_samples}/{self.total_samples}", refresh=True
            )

    def _set_view_postfix(self, *, refresh: bool = True) -> None:
        if self._views is not None:
            self._views.set_postfix_str(
                f"accepted={self.accepted}, rejected={self.rejected}, "
                f"existing={self.existing}",
                refresh=refresh,
            )

    def _close_views(self) -> None:
        if self._views is not None:
            self._views.close()
            self._views = None

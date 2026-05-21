from __future__ import annotations

from pathlib import Path

import fishs2_fastapi.model_loader as model_loader


def _fake_native_class(*, with_shared_init: bool, with_gpu_layer_export: bool):
    class FakeNative:
        def __init__(self, dll_path: Path):
            _ = dll_path
            self.calls: list[tuple[str, int | None]] = []
            self.initialize_audio_codec_model_shared = (
                self._initialize_audio_codec_model_shared if with_shared_init else None
            )
            self.initialize_model_with_gpu_layers = (
                self._initialize_model_with_gpu_layers if with_gpu_layer_export else None
            )

        def alloc_model(self):
            return object()

        def release_model(self, handle):
            _ = handle

        def alloc_audio_codec(self):
            return object()

        def release_audio_codec(self, handle):
            _ = handle

        def alloc_tokenizer(self):
            return object()

        def release_tokenizer(self, handle):
            _ = handle

        def alloc_pipeline(self):
            return object()

        def release_pipeline(self, handle):
            _ = handle

        def alloc_generate_params(self):
            return object()

        def release_generate_params(self, handle):
            _ = handle

        def alloc_audio_buffer(self, initial_size):
            _ = initial_size
            return object()

        def release_audio_buffer(self, handle):
            _ = handle

        def _initialize_audio_codec_model_shared(
            self,
            model_handle,
            codec_handle,
            model_path,
            gpu_device,
            backend_id,
        ) -> int:
            _ = (model_handle, codec_handle, model_path, gpu_device, backend_id)
            self.calls.append(("shared_init", None))
            return 1

        def _initialize_model_with_gpu_layers(
            self,
            model_handle,
            model_path,
            gpu_device,
            backend_id,
            n_gpu_layers,
        ) -> int:
            _ = (model_handle, model_path, gpu_device, backend_id)
            self.calls.append(("model_with_layers", int(n_gpu_layers)))
            return 1

        def initialize_model(self, model_handle, model_path, gpu_device, backend_id) -> int:
            _ = (model_handle, model_path, gpu_device, backend_id)
            self.calls.append(("model", None))
            return 1

        def initialize_audio_codec(self, codec_handle, model_path, gpu_device, backend_id) -> int:
            _ = (codec_handle, model_path, gpu_device, backend_id)
            self.calls.append(("codec", None))
            return 1

        def initialize_tokenizer(self, tokenizer_handle, tokenizer_path) -> int:
            _ = (tokenizer_handle, tokenizer_path)
            return 1

        def sync_tokenizer_config(self, model_handle, tokenizer_handle):
            _ = (model_handle, tokenizer_handle)

        def initialize_pipeline(
            self,
            pipeline_handle,
            tokenizer_handle,
            model_handle,
            codec_handle,
        ) -> int:
            _ = (pipeline_handle, tokenizer_handle, model_handle, codec_handle)
            return 1

        def initialize_generate_params(
            self,
            generate_params_handle,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            min_tokens_before_end,
            n_threads,
            verbose,
        ) -> int:
            _ = (
                generate_params_handle,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                min_tokens_before_end,
                n_threads,
                verbose,
            )
            return 1

    return FakeNative


def _patch_runtime_dependencies(monkeypatch, tmp_path: Path) -> None:
    model_file = tmp_path / "model.gguf"
    tokenizer_file = tmp_path / "tokenizer.json"
    dll_file = tmp_path / "s2.dll"

    model_file.write_bytes(b"model")
    tokenizer_file.write_bytes(b"tokenizer")
    dll_file.write_bytes(b"dll")

    def _resolve_artifact_path(path: Path, *, label: str) -> Path:
        _ = path
        if label == "model":
            return model_file
        return tokenizer_file

    monkeypatch.setattr(model_loader, "_resolve_artifact_path", _resolve_artifact_path)
    monkeypatch.setattr(model_loader, "resolve_s2_dll_path", lambda: dll_file)
    monkeypatch.setattr(model_loader, "bootstrap_dll_search_paths", lambda runtime_dir: None)
    monkeypatch.setattr(model_loader.settings, "backend", "cpu")
    monkeypatch.setattr(model_loader.settings, "gpu_device", 0)


def test_runtime_uses_n_gpu_layers_export_when_available(monkeypatch, tmp_path):
    _patch_runtime_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(model_loader.settings, "n_gpu_layers", 12)
    monkeypatch.setattr(
        model_loader,
        "_S2Native",
        _fake_native_class(with_shared_init=True, with_gpu_layer_export=True),
    )

    runtime = model_loader.S2Runtime("fishaudio/s2-pro")
    runtime._load()

    assert ("model_with_layers", 12) in runtime._native.calls
    assert ("shared_init", None) not in runtime._native.calls

    runtime.close()


def test_runtime_falls_back_when_n_gpu_layers_export_missing(monkeypatch, tmp_path):
    _patch_runtime_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(model_loader.settings, "n_gpu_layers", 8)
    monkeypatch.setattr(
        model_loader,
        "_S2Native",
        _fake_native_class(with_shared_init=True, with_gpu_layer_export=False),
    )

    runtime = model_loader.S2Runtime("fishaudio/s2-pro")
    runtime._load()

    assert ("model", None) in runtime._native.calls
    assert ("shared_init", None) not in runtime._native.calls

    runtime.close()


def test_runtime_uses_shared_init_when_n_gpu_layers_auto(monkeypatch, tmp_path):
    _patch_runtime_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(model_loader.settings, "n_gpu_layers", -1)
    monkeypatch.setattr(
        model_loader,
        "_S2Native",
        _fake_native_class(with_shared_init=True, with_gpu_layer_export=True),
    )

    runtime = model_loader.S2Runtime("fishaudio/s2-pro")
    runtime._load()

    assert ("shared_init", None) in runtime._native.calls
    assert ("model_with_layers", -1) not in runtime._native.calls

    runtime.close()

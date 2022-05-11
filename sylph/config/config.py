#!/usr/bin/env python3
import contextlib
import os

from unittest import mock
import pkg_resources
import yaml
from d2go.config import CfgNode as _CfgNode
from d2go.config import reroute_config_path as _reroute_config_path


class CfgNode(_CfgNode):
    def merge_from_file(self, cfg_filename: str, *args, **kwargs):
        cfg_filename = reroute_config_path(cfg_filename)
        with reroute_load_yaml_with_base():
            return super().merge_from_file(cfg_filename, *args, **kwargs)

    @staticmethod
    def load_yaml_with_base(filename: str, *args, **kwargs):
        with reroute_load_yaml_with_base():
            return _CfgNode.load_yaml_with_base(filename, *args, **kwargs)


def reroute_config_path(path: str) -> str:
    path = _reroute_config_path(path)

    if path.startswith("sylph://"):
        rel_path = path[len("sylph://") :]

        config_in_resource = pkg_resources.resource_filename(
            "sylph.model_zoo", os.path.join("configs", rel_path)
        )
        return config_in_resource
    return path


@contextlib.contextmanager
def reroute_load_yaml_with_base():
    BASE_KEY = "_BASE_"
    _safe_load = yaml.safe_load
    _unsafe_load = yaml.unsafe_load

    def mock_safe_load(f):
        cfg = _safe_load(f)
        if BASE_KEY in cfg:
            cfg[BASE_KEY] = reroute_config_path(cfg[BASE_KEY])
        return cfg

    def mock_unsafe_load(f):
        cfg = _unsafe_load(f)
        if BASE_KEY in cfg:
            cfg[BASE_KEY] = reroute_config_path(cfg[BASE_KEY])
        return cfg

    with mock.patch("yaml.safe_load", side_effect=mock_safe_load):
        with mock.patch("yaml.unsafe_load", side_effect=mock_unsafe_load):
            yield

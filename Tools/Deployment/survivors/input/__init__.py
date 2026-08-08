"""Survivors 用 focus-safe semantic input lease package。
controller と OS 入力を別 process に分離し、短い期限と対象照合で入力の取り残しを防ぎます。
"""
from .controller import HelperUnavailable, InputLeaseController
from .lease_protocol import Lease, LeaseValidator
__all__ = ["HelperUnavailable", "InputLeaseController", "Lease", "LeaseValidator"]

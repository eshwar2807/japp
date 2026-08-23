"""Driver registry: pick the right portal driver for a job URL."""

from __future__ import annotations

from automation.ats_drivers.base_driver import ApplicationOutcome, BaseATSDriver
from automation.ats_drivers.greenhouse_driver import GreenhouseDriver
from automation.ats_drivers.workday_driver import WorkdayDriver

#: Order matters only in that the first match wins.
DRIVERS: tuple[type[BaseATSDriver], ...] = (WorkdayDriver, GreenhouseDriver)


def get_driver_class(url: str) -> type[BaseATSDriver]:
    """Return the driver claiming this URL, falling back to the generic one."""
    for driver in DRIVERS:
        if driver.matches(url):
            return driver
    return BaseATSDriver


__all__ = [
    "ApplicationOutcome",
    "BaseATSDriver",
    "GreenhouseDriver",
    "WorkdayDriver",
    "DRIVERS",
    "get_driver_class",
]

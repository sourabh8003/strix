"""Tests for the scan-wide ``max_agents`` cap on ``spawn_child_agent``.

Dollar-cost budgeting (``max_budget_usd``) is a no-op for model-subscription
backends — their usage always reports as $0 cost — so agent count is the only
guardrail available there against a runaway "massive parallel swarm" burning
through a subscription's shared usage window.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from strix.core import execution
from strix.core.agents import AgentCoordinator


async def _register(coordinator: AgentCoordinator, count: int) -> None:
    for i in range(count):
        parent_id = None if i == 0 else "agent-0"
        await coordinator.register(f"agent-{i}", f"Agent {i}", parent_id)


def _factory_spy() -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return object()

    return calls, factory


@pytest.mark.asyncio
async def test_spawn_child_agent_refuses_once_cap_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = AgentCoordinator()
    await _register(coordinator, 3)  # root + 2 children already exist

    start_child_runner = AsyncMock()
    monkeypatch.setattr(execution, "_start_child_runner", start_child_runner)
    calls, factory = _factory_spy()

    result = await execution.spawn_child_agent(
        coordinator=coordinator,
        factory=factory,
        agents_db_path=None,  # type: ignore[arg-type]
        sessions_to_close=[],
        run_config=None,  # type: ignore[arg-type]
        max_turns=10,
        max_agents=3,
        interactive=False,
        parent_ctx={"agent_id": "agent-0"},
        name="New Specialist",
        task="probe something",
        skills=[],
        parent_history=[],
    )

    assert result["success"] is False
    assert "cap" in result["error"].lower()
    assert calls == []  # no child agent was built
    start_child_runner.assert_not_called()
    assert len(coordinator.statuses) == 3  # nothing new was registered


@pytest.mark.asyncio
async def test_spawn_child_agent_allows_when_under_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = AgentCoordinator()
    await _register(coordinator, 2)  # root + 1 child

    start_child_runner = AsyncMock()
    monkeypatch.setattr(execution, "_start_child_runner", start_child_runner)
    calls, factory = _factory_spy()

    result = await execution.spawn_child_agent(
        coordinator=coordinator,
        factory=factory,
        agents_db_path=None,  # type: ignore[arg-type]
        sessions_to_close=[],
        run_config=None,  # type: ignore[arg-type]
        max_turns=10,
        max_agents=3,
        interactive=False,
        parent_ctx={"agent_id": "agent-0"},
        name="New Specialist",
        task="probe something",
        skills=[],
        parent_history=[],
    )

    assert result["success"] is True
    assert len(calls) == 1
    start_child_runner.assert_awaited_once()
    assert len(coordinator.statuses) == 3  # the new child was registered


@pytest.mark.asyncio
async def test_spawn_child_agent_unbounded_when_max_agents_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = AgentCoordinator()
    await _register(coordinator, 50)

    start_child_runner = AsyncMock()
    monkeypatch.setattr(execution, "_start_child_runner", start_child_runner)
    calls, factory = _factory_spy()

    result = await execution.spawn_child_agent(
        coordinator=coordinator,
        factory=factory,
        agents_db_path=None,  # type: ignore[arg-type]
        sessions_to_close=[],
        run_config=None,  # type: ignore[arg-type]
        max_turns=10,
        max_agents=None,
        interactive=False,
        parent_ctx={"agent_id": "agent-0"},
        name="New Specialist",
        task="probe something",
        skills=[],
        parent_history=[],
    )

    assert result["success"] is True
    assert len(calls) == 1

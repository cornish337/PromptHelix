import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.services import create_experiment_run, add_chromosome_record
from prompthelix.genetics.engine import PromptChromosome


def setup_run_with_chromosomes(db: SQLAlchemySession, num_chromosomes: int = 3):
    run = create_experiment_run(db, parameters={"test": True})
    chromosomes = []
    for i in range(num_chromosomes):
        chromo = PromptChromosome(genes=[f"gene{i}"], fitness_score=float(i))
        add_chromosome_record(db, run, generation_number=i, chromosome=chromo)
        chromosomes.append(chromo)
    return run, chromosomes


def test_list_ga_experiment_runs(client: TestClient, db_session: SQLAlchemySession):
    run1 = create_experiment_run(db_session)
    run2 = create_experiment_run(db_session)

    response = client.get("/api/experiments/runs?skip=0&limit=10")
    assert response.status_code == 200
    data = response.json()
    ids = [r["id"] for r in data]
    assert run1.id in ids and run2.id in ids


def test_get_chromosomes_for_run(client: TestClient, db_session: SQLAlchemySession):
    run, chromos = setup_run_with_chromosomes(db_session, num_chromosomes=5)

    response = client.get(f"/api/experiments/runs/{run.id}/chromosomes?skip=1&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    returned_ids = [c["id"] for c in data]
    expected = [str(chromos[1].id), str(chromos[2].id)]
    assert returned_ids == expected


def test_get_chromosomes_for_unknown_run(client: TestClient):
    response = client.get("/api/experiments/runs/99999/chromosomes")
    assert response.status_code == 404

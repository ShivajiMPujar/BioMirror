"""End-to-end tests for authenticated Log Health flows and consumers."""

import os
import json
import sys
import tempfile
import unittest
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

_database_directory = tempfile.TemporaryDirectory(prefix="biomirror-log-health-")
_database_path = (Path(_database_directory.name) / "test.sqlite3").as_posix()
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_database_path}"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend_api as api
from database import PredictionHistoryRow


def mock_groq_response():
    async def handler(_request):
        return api.httpx.Response(200, json={
            "choices": [{"message": {"content": "• Test coach response."}}]
        })

    return api.httpx.MockTransport(handler)


class LogHealthEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await api._db.init()
        api.ANTHROPIC_API_KEY = ""

        suffix = uuid.uuid4().hex[:10]
        self.patient_id = f"P-{suffix.upper()}"
        self.username = f"logtest_{suffix}"
        await api._db.create_user({
            "username": self.username,
            "email": f"{self.username}@example.test",
            "password": "unused-test-hash",
            "full_name": "Log Health Test",
            "patient_id": self.patient_id,
            "role": "patient",
            "bmi": 25.1,
            "baseline_glucose": 140.0,
            "profile_complete": True,
        })
        await api._db.update_user(self.username, {
            "height_cm": 170.0,
            "weight_kg": 72.5,
            "bmi": 25.1,
            "hba1c": 7.2,
        })
        user = await api._db.get_user_by_username(self.username)
        await api.save_twin(self.patient_id, api.build_initial_twin(self.patient_id, user))

        self.client = AsyncClient(
            transport=ASGITransport(app=api.app),
            base_url="http://biomirror.test",
            headers={"Authorization": f"Bearer {api.make_access_token(self.username)}"},
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        api.TWIN_CACHE.pop(self.patient_id, None)
        api.TWIN_LAST_TICK.pop(self.patient_id, None)
        await api._db.close()
        _database_directory.cleanup()

    def mock_groq(self):
        previous_groq_key = api.GROQ_API_KEY
        previous_anthropic_key = api.ANTHROPIC_API_KEY
        self.addCleanup(setattr, api, "GROQ_API_KEY", previous_groq_key)
        self.addCleanup(setattr, api, "ANTHROPIC_API_KEY", previous_anthropic_key)
        api.GROQ_API_KEY = "test-groq-key"
        api.ANTHROPIC_API_KEY = ""

        original_client = api.httpx.AsyncClient
        transport = mock_groq_response()
        patcher = patch.object(
            api.httpx,
            "AsyncClient",
            side_effect=lambda *args, **kwargs: original_client(*args, transport=transport, **kwargs),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_all_categories_persist_and_reach_consumers(self):
        invalid_payloads = [
            ("glucose", {"glucose": 39}),
            ("meal", {"food_name": "apple", "quantity": 101}),
            ("activity", {"activity": "Walking", "duration_min": 481}),
            ("sleep", {"hours": 25}),
            ("water", {"amount_ml": 49}),
            ("weight", {"weight_kg": 19}),
            ("stress", {"level": "extreme"}),
            ("bp", {"systolic": 251, "diastolic": 80}),
            ("hba1c", {"hba1c": 21}),
            ("medication", {}),
        ]
        for endpoint, payload in invalid_payloads:
            with self.subTest(invalid_endpoint=endpoint):
                response = await self.client.post(
                    f"/logs/{self.patient_id}/{endpoint}", json=payload
                )
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(await api._db.count_logs(self.patient_id), 0)

        valid_payloads = [
            ("glucose", {"glucose": 145, "meal_type": "After meal"}, {"glucose": 145.0}),
            ("meal", {"food_name": "apple", "quantity": 1, "serving_unit": "piece", "meal_type": "Breakfast"}, {"meal_type": "Breakfast"}),
            ("activity", {"activity": "Running", "duration_min": 30, "intensity": "intense", "steps": 4000}, {"activity": "Running", "duration_min": 30, "steps": 4000}),
            ("sleep", {"hours": 7.5, "quality": 5}, {"hours": 7.5, "quality": 5}),
            ("water", {"amount_ml": 500}, {"amount_ml": 500}),
            ("weight", {"weight_kg": 71.8}, {"weight_kg": 71.8, "bmi": 24.8}),
            ("stress", {"level": "high"}, {"level": "high", "stress_level": 5}),
            ("bp", {"systolic": 125, "diastolic": 82}, {"systolic": 125, "diastolic": 82}),
            ("hba1c", {"hba1c": 6.9, "test_date": "2026-09-26"}, {"hba1c": 6.9}),
            ("medication", {"medication": "Metformin", "dose": "500 mg", "taken": True, "notes": "Frequency: Once daily"}, {"medication": "Metformin", "dose": "500 mg", "taken": True}),
        ]

        for index, (endpoint, payload, expected_fields) in enumerate(valid_payloads, start=1):
            with self.subTest(valid_endpoint=endpoint):
                response = await self.client.post(
                    f"/logs/{self.patient_id}/{endpoint}", json=payload
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json().get("success"))

                log_type = "blood_pressure" if endpoint == "bp" else endpoint
                stored = await api._db.get_logs(self.patient_id, log_type, limit=10)
                self.assertEqual(len(stored), 1)
                for field, expected in expected_fields.items():
                    self.assertEqual(stored[0].get(field), expected)

                history = await self.client.get(
                    f"/history/{self.patient_id}?log_type={log_type}&days=30&limit=200"
                )
                self.assertEqual(history.status_code, 200, history.text)
                self.assertEqual(len(history.json()["logs"]), 1)
                self.assertEqual(history.json()["logs"][0]["type"], log_type)

                dashboard = await self.client.get(f"/twin/{self.patient_id}")
                self.assertEqual(dashboard.status_code, 200, dashboard.text)
                self.assertIn("metrics", dashboard.json())

                insights = await self.client.get(f"/summary/{self.patient_id}/weekly")
                self.assertEqual(insights.status_code, 200, insights.text)
                self.assertEqual(insights.json()["logs_total"], index)

        history = await self.client.get(
            f"/history/{self.patient_id}?days=30&limit=200"
        )
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(
            {entry["type"] for entry in history.json()["logs"]},
            {"glucose", "meal", "activity", "sleep", "water", "weight", "stress", "blood_pressure", "hba1c", "medication"},
        )

        twin_response = await self.client.get(f"/twin/{self.patient_id}")
        twin = twin_response.json()
        expected_glucose = round(6.9 * 28.7 - 46.7, 1)
        self.assertAlmostEqual(twin["bergman_state"]["G"], expected_glucose, places=1)
        self.assertGreater(twin["bergman_state"]["p3_index"], 0)
        self.assertTrue(twin["history_length"] >= 0)
        self.assertGreaterEqual(twin["metrics"]["water_today_ml"], 500)

        summary = insights.json()
        self.assertEqual(summary["weight"]["latest_kg"], 71.8)
        self.assertEqual(summary["weight"]["latest_bmi"], 24.8)
        self.assertEqual(summary["medication"]["entries"], 1)
        self.assertEqual(summary["medication"]["taken_entries"], 1)
        self.assertIn("Metformin", summary["medication"]["names"])
        self.assertTrue(any("weight" in item.lower() for item in summary["insights"]))
        self.assertTrue(any("medication" in item.lower() for item in summary["insights"]))

        self.mock_groq()
        coach = await self.client.post("/coach/ask", json={
            "patient_id": self.patient_id,
            "message": "Summarize my recent logged data.",
            "include_context": True,
        })
        self.assertEqual(coach.status_code, 200, coach.text)
        coach_context = coach.json()["context_used"].lower()
        self.assertIn("recent meals: 1 piece apple", coach_context)
        self.assertIn("recent activity: 30 minutes", coach_context)
        self.assertIn("average sleep: 7.5 hours", coach_context)
        self.assertIn("latest weight: 71.8 kg", coach_context)

        simulation = await self.client.post("/simulate", json={
            "patient_id": self.patient_id,
            "carbs": 50,
            "exercise_min": 0,
            "sleep_hours": 7,
            "stress_level": 2,
        })
        self.assertEqual(simulation.status_code, 200, simulation.text)
        self.assertEqual(len(simulation.json()["baseline"]), 48)
        self.assertAlmostEqual(
            simulation.json()["results"]["p3_adjusted"],
            twin["bergman_state"]["p3_index"],
            places=1,
        )

    async def test_coach_uses_question_specific_context(self):
        self.mock_groq()
        for endpoint, payload in [
            ("glucose", {"glucose": 136, "meal_type": "Before meal"}),
            ("meal", {"food_name": "oatmeal", "quantity": 1, "serving_unit": "bowl", "meal_type": "Breakfast"}),
            ("activity", {"activity": "Walking", "duration_min": 20, "intensity": "moderate"}),
            ("sleep", {"hours": 7.2, "quality": 4}),
            ("weight", {"weight_kg": 72.1}),
        ]:
            response = await self.client.post(f"/logs/{self.patient_id}/{endpoint}", json=payload)
            self.assertEqual(response.status_code, 200, response.text)

        twin = await api.load_twin(self.patient_id)
        twin["G"] = 62.9
        await api.save_twin(self.patient_id, twin)

        food_question = await self.client.post("/coach/ask", json={
            "patient_id": self.patient_id,
            "message": "What should I eat for breakfast?",
            "include_context": True,
        })
        self.assertEqual(food_question.status_code, 200, food_question.text)
        food_context = food_question.json()["context_used"].lower()
        self.assertIn("recent meals", food_context)
        self.assertNotIn("reversal score", food_context)
        self.assertNotIn("latest glucose", food_context)

        reversal_question = await self.client.post("/coach/ask", json={
            "patient_id": self.patient_id,
            "message": "What is my reversal score right now?",
            "include_context": True,
        })
        self.assertEqual(reversal_question.status_code, 200, reversal_question.text)
        reversal_context = reversal_question.json()["context_used"].lower()
        self.assertIn("reversal score", reversal_context)
        self.assertIn("score formula", reversal_context)
        self.assertIn("time-in-range 35%", reversal_context)
        self.assertNotIn("recent meals", reversal_context)

        exercise_question = await self.client.post("/coach/ask", json={
            "patient_id": self.patient_id,
            "message": "What exercise should I do today?",
            "include_context": True,
        })
        self.assertEqual(exercise_question.status_code, 200, exercise_question.text)
        exercise_context = exercise_question.json()["context_used"].lower()
        self.assertIn("activity", exercise_context)
        self.assertNotIn("reversal score", exercise_context)

        glucose_question = await self.client.post("/coach/ask", json={
            "patient_id": self.patient_id,
            "message": "How is my glucose trending?",
            "include_context": True,
        })
        self.assertEqual(glucose_question.status_code, 200, glucose_question.text)
        glucose_context = glucose_question.json()["context_used"].lower()
        self.assertIn("latest glucose reading: 136.0 mg/dl", glucose_context)
        self.assertNotIn("62.9", glucose_context)
        self.assertNotIn("reversal score", glucose_context)

    async def test_groq_receives_current_question_without_prior_answers(self):
        await api._db.add_chat_message(
            self.patient_id, "assistant", "Stale answer: your glucose is 999 mg/dL."
        )
        captured_payloads = []

        async def groq_handler(request):
            payload = json.loads(request.content)
            captured_payloads.append(payload)
            current_question = payload["messages"][-1]["content"]
            return api.httpx.Response(200, json={
                "choices": [{"message": {"content": f"Direct answer: {current_question}"}}]
            })

        transport = api.httpx.MockTransport(groq_handler)
        original_client = api.httpx.AsyncClient
        previous_groq_key = api.GROQ_API_KEY
        previous_anthropic_key = api.ANTHROPIC_API_KEY
        api.GROQ_API_KEY = "test-groq-key"
        api.ANTHROPIC_API_KEY = ""
        questions = [
            "What exercise lowers glucose fastest?",
            "How is my glucose trending?",
            "Explain my reversal score.",
            "What foods should I choose?",
            "What can you help me with?",
        ]

        def mock_async_client(*args, **kwargs):
            return original_client(*args, transport=transport, **kwargs)

        try:
            with patch.object(api.httpx, "AsyncClient", side_effect=mock_async_client):
                responses = []
                for question in questions:
                    response = await self.client.post("/coach/ask", json={
                        "patient_id": self.patient_id,
                        "message": question,
                        "include_context": True,
                    })
                    self.assertEqual(response.status_code, 200, response.text)
                    responses.append(response.json())
        finally:
            api.GROQ_API_KEY = previous_groq_key
            api.ANTHROPIC_API_KEY = previous_anthropic_key

        self.assertEqual(len(captured_payloads), len(questions))
        for question, payload, response in zip(questions, captured_payloads, responses):
            with self.subTest(question=question):
                self.assertEqual(payload["messages"][-1], {"role": "user", "content": question})
                self.assertEqual([item["role"] for item in payload["messages"]], ["system", "user"])
                self.assertEqual(payload["reasoning_effort"], "low")
                self.assertEqual(payload["max_tokens"], 500)
                self.assertNotIn("Stale answer", payload["messages"][0]["content"])
                self.assertNotIn("glucose is 999 mg/dL", payload["messages"][0]["content"])
                self.assertEqual(response["ai_response"], f"Direct answer: {question}")
                self.assertEqual(response["model"], api.GROQ_MODEL)
                if question == "How can I build a consistent daily routine?":
                    self.assertNotIn("current glucose:", payload["messages"][0]["content"].lower())
                    self.assertNotIn("reversal score:", payload["messages"][0]["content"].lower())

    async def test_unavailable_ai_provider_does_not_return_canned_health_answer(self):
        previous_groq_key = api.GROQ_API_KEY
        previous_anthropic_key = api.ANTHROPIC_API_KEY
        api.GROQ_API_KEY = ""
        api.ANTHROPIC_API_KEY = ""
        try:
            response = await self.client.post("/coach/ask", json={
                "patient_id": self.patient_id,
                "message": "How is my glucose?",
                "include_context": True,
            })
        finally:
            api.GROQ_API_KEY = previous_groq_key
            api.ANTHROPIC_API_KEY = previous_anthropic_key

        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("no AI provider returned a response", response.json()["detail"])

    async def test_goals_can_be_removed_individually(self):
        saved = await self.client.post(f"/goals/{self.patient_id}", json={
            "target_hba1c": 6.2,
            "target_weight": 68.0,
            "target_steps": 8000,
            "target_date": "2027-01-01",
        })
        self.assertEqual(saved.status_code, 200, saved.text)

        removed = await self.client.delete(f"/goals/{self.patient_id}/target_weight")
        self.assertEqual(removed.status_code, 200, removed.text)
        self.assertNotIn("target_weight", removed.json()["goals"])
        self.assertEqual(removed.json()["goals"]["target_hba1c"], 6.2)
        self.assertEqual(removed.json()["goals"]["target_steps"], 8000)

        current = await self.client.get(f"/goals/{self.patient_id}")
        self.assertEqual(current.status_code, 200, current.text)
        self.assertNotIn("target_weight", current.json()["goals"])
        self.assertIn("target_hba1c", current.json()["goals"])

        invalid = await self.client.delete(f"/goals/{self.patient_id}/patient_id")
        self.assertEqual(invalid.status_code, 404, invalid.text)
        forbidden = await self.client.delete("/goals/P-OTHER/target_hba1c")
        self.assertEqual(forbidden.status_code, 403, forbidden.text)

    async def test_global_forecast_actual_error_and_personalized_next_forecast(self):
        initial_state = await api.load_twin(self.patient_id)
        global_response = await self.client.get(f"/forecast/{self.patient_id}?hours=1")
        self.assertEqual(global_response.status_code, 200, global_response.text)
        global_prediction = global_response.json()["near_term_prediction"]["global_predicted_glucose_mg_dl"]
        self.assertTrue(global_prediction)
        self.assertEqual(global_response.json()["personalization"]["error_samples"], 0)

        stored_forecasts = await api._db.get_predictions(self.patient_id, limit=10, source="forecast")
        matched_prediction = stored_forecasts[-1]
        async with api._db.Session() as session:
            row = (await session.execute(
                select(PredictionHistoryRow).where(PredictionHistoryRow.id == matched_prediction["id"])
            )).scalar_one()
            row.timestamp -= timedelta(minutes=5)
            await session.commit()

        observed_value = 140.0
        observed = await self.client.post(f"/logs/{self.patient_id}/glucose", json={
            "glucose": observed_value,
            "meal_type": "Fasting",
        })
        self.assertEqual(observed.status_code, 200, observed.text)
        feedback = observed.json()["personalization_update"]
        expected_error = round(observed_value - global_prediction[0], 2)
        self.assertAlmostEqual(feedback["inputs"]["error_mg_dl"], expected_error, places=2)
        self.assertEqual(observed.json()["current_state"]["glucose"], observed_value)

        anchored_state = await api.load_twin(self.patient_id)
        self.assertEqual(anchored_state["G"], observed_value)
        self.assertEqual(anchored_state["sim_time"], initial_state["sim_time"])

        next_response = await self.client.get(f"/forecast/{self.patient_id}?hours=1")
        self.assertEqual(next_response.status_code, 200, next_response.text)
        next_forecast = next_response.json()
        expected_correction = float(feedback["inputs"]["error_mg_dl"])
        self.assertEqual(next_forecast["personalization"]["correction_mg_dl"], expected_correction)
        raw_next = next_forecast["near_term_prediction"]["global_predicted_glucose_mg_dl"]
        corrected_next = next_forecast["near_term_prediction"]["predicted_glucose_mg_dl"]
        self.assertEqual(corrected_next, [round(value + expected_correction, 1) for value in raw_next])

        after_forecast_state = await api.load_twin(self.patient_id)
        self.assertEqual(after_forecast_state["G"], observed_value)
        feedback_rows = await api._db.get_predictions(self.patient_id, limit=10, source="feedback")
        self.assertEqual(len(feedback_rows), 1)

    async def test_registration_does_not_fabricate_glucose_baseline(self):
        suffix = uuid.uuid4().hex[:10]
        registration = await api.register(api.UserRegister(
            username=f"nobaseline_{suffix}",
            email=f"nobaseline_{suffix}@example.test",
            password="unused-test-password",
            full_name="No Baseline Test",
        ))
        patient_id = registration["patient_id"]
        self.assertIsNone(await api._db.get_twin_state(patient_id))
        registered_user = await api._db.get_user_by_username(f"nobaseline_{suffix}")
        self.assertIsNone(registered_user.get("baseline_glucose"))

        client = AsyncClient(
            transport=ASGITransport(app=api.app),
            base_url="http://biomirror.test",
            headers={"Authorization": f"Bearer {registration['access_token']}"},
        )
        try:
            analytics = await client.get(f"/analytics/{patient_id}")
            self.assertEqual(analytics.status_code, 200, analytics.text)
            self.assertIsNone(analytics.json()["glucose"]["mean"])
            self.assertIsNone(analytics.json()["glucose"]["tir_pct"])
            self.assertIsNone(analytics.json()["glucose"]["hba1c"])

            response = await client.post(f"/logs/{patient_id}/glucose", json={"glucose": 141})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["current_state"]["glucose"], 141.0)
            state = await api.load_twin(patient_id)
            self.assertEqual(state["G"], 141.0)
            self.assertEqual(state["history"][-1]["source"], "observed")
        finally:
            await client.aclose()
            api.TWIN_CACHE.pop(patient_id, None)
            api.TWIN_LAST_TICK.pop(patient_id, None)

    async def test_observed_glucose_anchors_twin_and_updates_user_correction(self):
        state_before = await api.load_twin(self.patient_id)
        observed_value = 140.0
        observed = await self.client.post(f"/logs/{self.patient_id}/glucose", json={
            "glucose": observed_value,
            "meal_type": "Fasting",
        })
        self.assertEqual(observed.status_code, 200, observed.text)
        feedback = observed.json()["personalization_update"]
        self.assertIsNotNone(feedback)
        stored_forecasts = await api._db.get_predictions(self.patient_id, limit=10, source="forecast")
        prediction = next(item for item in stored_forecasts if item["id"] == feedback["inputs"]["prediction_id"])
        expected_error = round(observed_value - prediction["predicted_glucose"][0], 2)
        self.assertAlmostEqual(feedback["inputs"]["error_mg_dl"], expected_error, places=2)
        self.assertEqual(prediction["inputs"]["personalization_correction_mg_dl"], 0.0)
        self.assertEqual(observed.json()["current_state"]["glucose"], observed_value)

        state_after_log = await api.load_twin(self.patient_id)
        self.assertEqual(state_after_log["G"], observed_value)
        self.assertEqual(state_after_log["sim_time"], state_before["sim_time"])

        personalization = await api.get_prediction_personalization(self.patient_id)
        self.assertEqual(personalization["correction_mg_dl"], expected_error)
        self.assertEqual(personalization["sample_count"], 1)
        self.assertEqual(await api.get_prediction_personalization("P-OTHER"), {
            "correction_mg_dl": 0.0,
            "sample_count": 0,
            "recent_errors_mg_dl": [],
        })

        forecast = await self.client.get(f"/forecast/{self.patient_id}?hours=1")
        self.assertEqual(forecast.status_code, 200, forecast.text)
        result = forecast.json()
        self.assertEqual(result["personalization"]["correction_mg_dl"], expected_error)
        raw = result["near_term_prediction"]["global_predicted_glucose_mg_dl"]
        personalized = result["near_term_prediction"]["predicted_glucose_mg_dl"]
        self.assertEqual(personalized, [round(value + expected_error, 1) for value in raw])

        second_forecast = await self.client.get(f"/forecast/{self.patient_id}?hours=1")
        self.assertEqual(second_forecast.status_code, 200, second_forecast.text)
        self.assertEqual(second_forecast.json()["personalization"]["correction_mg_dl"], expected_error)
        feedback_rows = await api._db.get_predictions(self.patient_id, limit=10, source="feedback")
        self.assertEqual(len(feedback_rows), 1)

    def test_real_time_ode_step_uses_elapsed_minutes(self):
        engine = api.BergmanEngine(p3=3.5e-5)
        glucose_after, _ = engine.step(140.0, 0.02, 12.0, 0.0, dt=2 / 60)
        self.assertAlmostEqual(glucose_after, 139.86, places=2)
        self.assertLess(abs(glucose_after - 140.0), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
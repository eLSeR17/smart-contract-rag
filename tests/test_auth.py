"""Unit tests for the API-key auth store."""

from __future__ import annotations

from smart_contract_rag.auth import APIKeyStore


class TestAPIKeyStore:
    def test_create_and_validate(self) -> None:
        store = APIKeyStore(db_path=":memory:")
        key_id, plaintext = store.create_key("alice")
        assert key_id.startswith("scrag_")
        assert len(plaintext) >= 32
        assert store.validate_key(plaintext) == key_id
        store.close()

    def test_plaintext_not_recoverable(self) -> None:
        store = APIKeyStore(db_path=":memory:")
        _, plaintext = store.create_key("alice")
        # Simulate a new store over the same data: only the hash survives.
        store2 = APIKeyStore(db_path=":memory:")
        for row in store._db().execute("SELECT * FROM api_keys").fetchall():
            store2._db().execute(
                "INSERT INTO api_keys (id, name, key_hash, rate_limit_per_min, created_at, active) "
                "VALUES (?, ?, ?, ?, ?, ?)", row
            )
        store2._db().commit()
        assert store2.validate_key(plaintext) is not None
        assert store2.validate_key("wrong-key") is None
        store.close()
        store2.close()

    def test_invalid_key_rejected(self) -> None:
        store = APIKeyStore(db_path=":memory:")
        assert store.validate_key("") is None
        assert store.validate_key("nonsense") is None
        store.close()

    def test_revoked_key_rejected(self) -> None:
        store = APIKeyStore(db_path=":memory:")
        key_id, plaintext = store.create_key("bob")
        store.revoke_key(key_id)
        assert store.validate_key(plaintext) is None
        store.close()

    def test_rate_limit_per_key(self) -> None:
        store = APIKeyStore(db_path=":memory:")
        _, _ = store.create_key("a", rate_limit_per_min=10)
        key_id_b, _ = store.create_key("b", rate_limit_per_min=30)
        assert store.rate_limit_for(key_id_b) == 30
        store.close()

    def test_list_keys_no_plaintext(self) -> None:
        store = APIKeyStore(db_path=":memory:")
        _, _ = store.create_key("alice")
        keys = store.list_keys()
        assert len(keys) == 1
        assert "key_hash" not in keys[0]
        assert keys[0]["name"] == "alice"
        store.close()

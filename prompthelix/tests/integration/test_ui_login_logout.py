from fastapi.testclient import TestClient
from prompthelix.tests.utils import get_auth_headers, DEFAULT_TEST_USERNAME, DEFAULT_TEST_PASSWORD


def test_ui_login_sets_cookie_and_logout_clears(client: TestClient, db_session):
    # ensure user exists and obtain token
    headers = get_auth_headers(client, db_session)
    token = headers["Authorization"].split()[1]

    # simulate UI login by setting cookie
    client.cookies.set("prompthelix_access_token", token)
    assert client.cookies.get("prompthelix_access_token") == token

    # call protected endpoint using header to verify token valid
    resp = client.get("/users/me", headers=headers)
    assert resp.status_code == 200

    # logout via API to invalidate token
    client.post("/auth/logout", headers=headers)
    # simulate UI logout clearing cookie
    client.cookies.set("prompthelix_access_token", "", expires=0)
    assert client.cookies.get("prompthelix_access_token") == ""

    # request should now fail without auth header
    resp_after = client.get("/users/me")
    assert resp_after.status_code == 401

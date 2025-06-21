print(">>> MINIMAL MAIN.PY EXECUTION STARTED <<<")
from fastapi import FastAPI, Request  # Request might not be needed here, but good for consistency
from fastapi.responses import JSONResponse

app = FastAPI()
print(">>> FastAPI app object created in MINIMAL main.py <<<")


@app.get("/")
async def root():
    return {"message": "Welcome to Minimal PromptHelix API"}


@app.get("/debug-routes")
async def debug_routes():
    """
    Temporary endpoint to list all registered routes for debugging.
    """
    routes = []
    for route in app.routes:
        route_info = {
            "path": "N/A",
            "name": "N/A",
            "methods": []
        }
        if hasattr(route, "path"):
            route_info["path"] = route.path
        if hasattr(route, "name"):
            route_info["name"] = route.name
        if hasattr(route, "methods"):
            route_info["methods"] = sorted(list(route.methods))
        routes.append(route_info)
    return JSONResponse(content={"routes": routes})


print(">>> MINIMAL MAIN.PY EXECUTION COMPLETED TO THE END <<<")

"""Tools the agent can call. Server tools run here; device tools are forwarded to the phone or browser."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    where: str  # "server" or "device"
    side_effect: bool = False
    run: Callable[..., Awaitable[dict]] | None = None  # server tools only

    def spec(self, api: str) -> dict:
        fn = {"name": self.name, "description": self.description, "parameters": self.parameters}
        if api == "chat":
            return {"type": "function", "function": fn}
        return {"type": "function", "strict": False, **fn}


def obj(props: dict, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


def http_client() -> httpx.AsyncClient:  # tests swap this for a MockTransport client
    return httpx.AsyncClient(timeout=8)


# WMO weather interpretation codes, as Open-Meteo reports them
WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers",
       81: "showers", 82: "heavy showers", 85: "snow showers", 86: "heavy snow showers",
       95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with heavy hail"}


def _c(f: float | None) -> int | None:
    return None if f is None else round((f - 32) * 5 / 9)


def summarize_forecast(d: dict, place: str | None) -> dict:
    """Open-Meteo's response cut down to what a spoken answer needs (keeps tool output tokens small)."""
    cur, hourly, daily = d["current"], d["hourly"], d["daily"]
    start = next((i for i, t in enumerate(hourly["time"]) if t >= cur["time"][:13]), 0)
    next_hours = [{"time": hourly["time"][i][11:16], "temp_f": round(hourly["temperature_2m"][i]),
                   "rain_chance_pct": hourly["precipitation_probability"][i],
                   "sky": WMO.get(hourly["weather_code"][i], "unknown")}
                  for i in range(start, min(start + 13, len(hourly["time"])), 3)]
    days = [{"date": daily["time"][i], "high_f": round(daily["temperature_2m_max"][i]),
             "low_f": round(daily["temperature_2m_min"][i]),
             "high_c": _c(daily["temperature_2m_max"][i]), "low_c": _c(daily["temperature_2m_min"][i]),
             "rain_chance_pct": daily["precipitation_probability_max"][i],
             "sky": WMO.get(daily["weather_code"][i], "unknown")} for i in range(min(2, len(daily["time"])))]
    return {"place": place,
            "now": {"temp_f": round(cur["temperature_2m"]), "temp_c": _c(cur["temperature_2m"]),
                    "feels_like_f": round(cur["apparent_temperature"]),
                    "sky": WMO.get(cur["weather_code"], "unknown"), "wind_mph": round(cur["wind_speed_10m"])},
            "next_12h": next_hours, "today": days[0] if days else None,
            "tomorrow": days[1] if len(days) > 1 else None}


async def get_weather(place: str | None = None, latitude: float | None = None,
                      longitude: float | None = None) -> dict:
    async with http_client() as http:
        if latitude is None or longitude is None:
            if not place:
                return {"error": "need a place name, or latitude and longitude from get_location"}
            r = await http.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": place, "count": 1})
            hits = r.json().get("results") or []
            if not hits:
                return {"error": f"couldn't find a place called {place!r}"}
            h = hits[0]
            latitude, longitude = h["latitude"], h["longitude"]
            place = ", ".join(x for x in (h.get("name"), h.get("admin1") or h.get("country")) if x)
        r = await http.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": latitude, "longitude": longitude, "timezone": "auto", "forecast_days": 2,
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "hourly": "temperature_2m,precipitation_probability,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"})
        r.raise_for_status()
        return summarize_forecast(r.json(), place)


SERVER_TOOLS = {
    "get_weather": Tool(
        "get_weather",
        "Weather now and the forecast through tomorrow. Pass a place name, or latitude and longitude "
        "from get_location when the user means where they are.",
        obj({"place": {"type": "string"}, "latitude": {"type": "number"}, "longitude": {"type": "number"}}),
        "server", run=get_weather),
}

DEVICE_TOOLS = {
    "get_location": Tool(
        "get_location",
        "The user's current location: coordinates and, when the phone knows it, a place name. Call it "
        "first when the user asks about weather or anything nearby without naming a place.",
        obj({}), "device"),
    "create_reminder": Tool(
        "create_reminder",
        "Add a reminder on the user's phone. `due` is local time in ISO 8601 (for example "
        "2026-11-03T18:00); leave it out when the user gave no time.",
        obj({"title": {"type": "string", "description": "What to do, as a short phrase that starts with a verb, "
                                                        "like 'Buy coffee filters' or 'Go to the night market'."},
             "due": {"type": "string"}, "notes": {"type": "string"}}, ("title",)),
        "device", side_effect=True),
}


def available(device_tools: list[str]) -> dict[str, Tool]:
    """Server tools plus the device tools this client says it can run."""
    return {**SERVER_TOOLS, **{n: DEVICE_TOOLS[n] for n in device_tools if n in DEVICE_TOOLS}}

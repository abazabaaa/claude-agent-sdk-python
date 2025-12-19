#!/usr/bin/env python3
"""
Simple weather script using wttr.in
====================================

Usage:
    python weather.py                    # Current location
    python weather.py "New York"         # Specific city
    python weather.py "London" --detailed # Full forecast
"""

import argparse
import urllib.request
import urllib.error
import urllib.parse


def get_weather(location: str = "", detailed: bool = False) -> str:
    """
    Fetch weather from wttr.in.

    Args:
        location: City name or empty for auto-detect
        detailed: If True, show detailed forecast

    Returns:
        Weather information as string
    """
    base_url = "https://wttr.in"

    # URL encode the location
    encoded_location = urllib.parse.quote(location) if location else ""

    if detailed:
        url = f"{base_url}/{encoded_location}"
    else:
        url = f"{base_url}/{encoded_location}?format=4"

    # wttr.in gives nicer output for curl user-agent
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "curl/7.68.0",
            "Accept": "text/plain",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as e:
        return f"Error fetching weather: {e}"
    except Exception as e:
        return f"Error: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get weather information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python weather.py                     # Auto-detect location
    python weather.py "San Francisco"     # Specific city
    python weather.py "Tokyo" --detailed  # Full forecast
    python weather.py "Paris,France"      # City with country
        """,
    )

    parser.add_argument(
        "location",
        nargs="?",
        default="",
        help="City name (default: auto-detect from IP)",
    )

    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed 3-day forecast",
    )

    args = parser.parse_args()

    weather = get_weather(args.location, args.detailed)
    print(weather)


if __name__ == "__main__":
    main()

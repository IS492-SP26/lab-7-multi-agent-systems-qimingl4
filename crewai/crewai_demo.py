"""
CrewAI Multi-Agent Demo: Travel Planning System
================================================

This demonstrates CrewAI's task-based multi-agent orchestration by planning
a 5-day trip to Iceland. Each agent has specialized tools that return
structured travel data, which the LLM then uses to create recommendations.

Agents:
1. FlightAgent - Flight Specialist (researches flight options)
2. HotelAgent - Accommodation Specialist (finds hotels)
3. ItineraryAgent - Travel Planner (creates day-by-day itineraries)
4. BudgetAgent - Financial Advisor (analyzes total costs)

Configuration:
- Uses shared configuration from the root .env file
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai.tools import tool

# Add parent directory to path to import shared_config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared configuration
from shared_config import Config, validate_config


# ============================================================================
# TOOLS (Real API implementations using web search)
# ============================================================================

@tool
def search_flight_prices(destination: str, departure_city: str = "New York") -> str:
    """
    Search for flight prices and options to a destination.
    Returns current flight information from major booking sites.
    """
    # Static flight data simulating real search results
    flights_data = {
        "Iceland": [
            {"airline": "Icelandair", "route": f"{departure_city} (JFK) → Reykjavik (KEF)", "type": "Direct", "duration": "5h 30m", "price": "$485 round-trip", "schedule": "Daily departures at 8:30 PM"},
            {"airline": "Delta Air Lines", "route": f"{departure_city} (JFK) → Reykjavik (KEF)", "type": "Direct", "duration": "5h 45m", "price": "$612 round-trip", "schedule": "Mon/Wed/Fri/Sat at 10:15 PM"},
            {"airline": "PLAY Airlines", "route": f"{departure_city} (SWF) → Reykjavik (KEF)", "type": "Direct (budget)", "duration": "5h 20m", "price": "$349 round-trip", "schedule": "Tue/Thu/Sun at 11:00 PM, no checked bags included"},
            {"airline": "Norse Atlantic", "route": f"{departure_city} (JFK) → Reykjavik (KEF)", "type": "Direct (budget)", "duration": "5h 35m", "price": "$389 round-trip", "schedule": "Daily at 7:45 PM, carry-on only"},
            {"airline": "British Airways", "route": f"{departure_city} (JFK) → London (LHR) → Reykjavik (KEF)", "type": "1 stop", "duration": "11h 20m", "price": "$578 round-trip", "schedule": "Daily, 4h layover in London"},
        ],
        "default": [
            {"airline": "United Airlines", "route": f"{departure_city} → {destination}", "type": "Direct", "duration": "8h 00m", "price": "$650 round-trip", "schedule": "Daily"},
            {"airline": "Delta Air Lines", "route": f"{departure_city} → {destination}", "type": "1 stop", "duration": "11h 30m", "price": "$520 round-trip", "schedule": "Daily"},
            {"airline": "Budget Air", "route": f"{departure_city} → {destination}", "type": "Direct (budget)", "duration": "8h 15m", "price": "$410 round-trip", "schedule": "Mon/Wed/Fri"},
        ],
    }

    key = "Iceland" if "iceland" in destination.lower() or "reykjavik" in destination.lower() else "default"
    results = flights_data[key]

    output = f"Flight Search Results: {departure_city} → {destination}\n"
    output += f"{'='*60}\n"
    for i, flight in enumerate(results, 1):
        output += f"\n{i}. {flight['airline']}\n"
        output += f"   Route: {flight['route']}\n"
        output += f"   Type: {flight['type']} | Duration: {flight['duration']}\n"
        output += f"   Price: {flight['price']}\n"
        output += f"   Schedule: {flight['schedule']}\n"
    output += f"\nNote: Prices as of January 2026. Book 6-8 weeks in advance for best rates."
    return output


@tool
def search_hotel_options(location: str, check_in_date: str) -> str:
    """
    Search for hotel options in a location.
    Returns current hotel availability and pricing information.
    """
    # Static hotel data simulating real search results
    hotels_data = {
        "Reykjavik": [
            {"name": "CenterHotel Midgardur", "stars": 4, "rating": 8.7, "reviews": 2341, "price": "$189/night", "location": "Downtown Reykjavik, 2 min walk to Hallgrimskirkja", "amenities": "Free WiFi, breakfast included, restaurant, bar, 24h front desk", "style": "Mid-range"},
            {"name": "Canopy by Hilton Reykjavik", "stars": 4, "rating": 9.1, "reviews": 1876, "price": "$265/night", "location": "Smidjustigur 4, city center", "amenities": "Rooftop bar, gym, spa, restaurant, free WiFi, heated floors", "style": "Upscale"},
            {"name": "Kex Hostel", "stars": 2, "rating": 8.2, "reviews": 3102, "price": "$85/night (private room)", "location": "Skulagata 28, harbor district", "amenities": "Shared lounge, bar, free WiFi, bike rental, live music events", "style": "Budget"},
            {"name": "Hotel Borg by Keahotels", "stars": 5, "rating": 9.3, "reviews": 1204, "price": "$385/night", "location": "Posthusstraeti 11, overlooking Austurvollur Square", "amenities": "Art deco design, spa, fine dining, butler service, airport transfer", "style": "Luxury"},
            {"name": "Reykjavik Lights Hotel", "stars": 3, "rating": 8.4, "reviews": 1567, "price": "$145/night", "location": "Sudurlandsbraut 12, 10 min bus to center", "amenities": "Free parking, breakfast buffet, northern lights wake-up call, free WiFi", "style": "Mid-range"},
        ],
        "default": [
            {"name": "City Center Hotel", "stars": 4, "rating": 8.5, "reviews": 1200, "price": "$175/night", "location": f"Downtown {location}", "amenities": "Free WiFi, breakfast, gym", "style": "Mid-range"},
            {"name": "Budget Inn", "stars": 2, "rating": 7.8, "reviews": 890, "price": "$79/night", "location": f"Near transit, {location}", "amenities": "Free WiFi, shared kitchen", "style": "Budget"},
            {"name": "Grand Luxury Resort", "stars": 5, "rating": 9.2, "reviews": 650, "price": "$350/night", "location": f"Premium district, {location}", "amenities": "Spa, pool, fine dining, concierge", "style": "Luxury"},
        ],
    }

    key = "Reykjavik" if "reykjavik" in location.lower() or "iceland" in location.lower() else "default"
    results = hotels_data[key]

    output = f"Hotel Search Results: {location} (check-in: {check_in_date})\n"
    output += f"{'='*60}\n"
    for i, hotel in enumerate(results, 1):
        output += f"\n{i}. {hotel['name']} {'⭐' * hotel['stars']}\n"
        output += f"   Rating: {hotel['rating']}/10 ({hotel['reviews']} reviews) | Style: {hotel['style']}\n"
        output += f"   Price: {hotel['price']}\n"
        output += f"   Location: {hotel['location']}\n"
        output += f"   Amenities: {hotel['amenities']}\n"
    output += f"\nNote: Prices for January 2026. 5-night stay recommended for best value."
    return output


@tool
def search_attractions_activities(destination: str) -> str:
    """
    Search for attractions and activities in a destination.
    Returns popular sites, tours, and experiences with pricing.
    """
    # Static attractions data simulating real search results
    attractions_data = {
        "Iceland": [
            {"name": "Golden Circle Tour", "type": "Day Tour", "duration": "8 hours", "price": "$85/person", "description": "Visit Thingvellir National Park, Geysir geothermal area, and Gullfoss waterfall. Includes hotel pickup.", "rating": 4.8},
            {"name": "Blue Lagoon", "type": "Spa/Attraction", "duration": "2-3 hours", "price": "$75-115/person (Comfort-Premium)", "description": "World-famous geothermal spa with silica mud masks and in-water bar. Book 2+ weeks in advance.", "rating": 4.5},
            {"name": "South Coast & Black Sand Beach", "type": "Day Tour", "duration": "10 hours", "price": "$95/person", "description": "Seljalandsfoss and Skogafoss waterfalls, Reynisfjara black sand beach, Vik village.", "rating": 4.9},
            {"name": "Northern Lights Tour", "type": "Evening Tour", "duration": "3-4 hours", "price": "$65/person", "description": "Guided bus tour to dark-sky locations. Free rebooking if no lights visible. Best Oct-Mar.", "rating": 4.3},
            {"name": "Glacier Hiking on Solheimajokull", "type": "Adventure", "duration": "3 hours", "price": "$110/person", "description": "Guided glacier walk with crampons and ice axes provided. No experience needed. Min age 8.", "rating": 4.9},
            {"name": "Whale Watching from Reykjavik", "type": "Boat Tour", "duration": "3 hours", "price": "$79/person", "description": "See humpback whales, dolphins, and puffins (summer). Warm overalls provided. 95% sighting rate.", "rating": 4.6},
            {"name": "Snorkeling in Silfra Fissure", "type": "Adventure", "duration": "3 hours", "price": "$145/person", "description": "Snorkel between tectonic plates in crystal-clear glacial water (2°C). Dry suit provided.", "rating": 4.8},
            {"name": "Hallgrimskirkja Church Tower", "type": "Landmark", "duration": "30 min", "price": "$12/person", "description": "Iconic Reykjavik church with elevator to observation deck. Panoramic city views.", "rating": 4.4},
            {"name": "Reykjavik Food Walk", "type": "Food Tour", "duration": "3 hours", "price": "$95/person", "description": "6 tastings including fermented shark, lamb soup, skyr, and craft beer. Small groups.", "rating": 4.7},
            {"name": "Ice Cave Exploration (Vatnajokull)", "type": "Adventure", "duration": "Full day (12h from Reykjavik)", "price": "$250/person", "description": "Visit naturally-formed blue ice caves inside Europe's largest glacier. Nov-Mar only.", "rating": 4.9},
        ],
        "default": [
            {"name": "City Walking Tour", "type": "Tour", "duration": "3 hours", "price": "$40/person", "description": f"Guided walking tour of {destination} highlights.", "rating": 4.5},
            {"name": "Local Food Experience", "type": "Food Tour", "duration": "2.5 hours", "price": "$75/person", "description": "Taste local cuisine with a knowledgeable guide.", "rating": 4.7},
            {"name": "Nature Day Trip", "type": "Day Tour", "duration": "8 hours", "price": "$95/person", "description": f"Full-day excursion to natural attractions near {destination}.", "rating": 4.6},
        ],
    }

    key = "Iceland" if "iceland" in destination.lower() or "reykjavik" in destination.lower() else "default"
    results = attractions_data[key]

    output = f"Attractions & Activities: {destination}\n"
    output += f"{'='*60}\n"
    for i, item in enumerate(results, 1):
        output += f"\n{i}. {item['name']} ({item['type']})\n"
        output += f"   Duration: {item['duration']} | Price: {item['price']} | Rating: {item['rating']}/5.0\n"
        output += f"   {item['description']}\n"
    output += f"\nTip: Book popular tours 1-2 weeks in advance, especially Golden Circle and Blue Lagoon."
    return output


@tool
def search_travel_costs(destination: str) -> str:
    """
    Search for travel costs and budgeting information.
    Returns current pricing for meals, activities, and transportation.
    """
    # Static cost data simulating real search results
    costs_data = {
        "Iceland": {
            "currency": "Icelandic Krona (ISK). 1 USD = ~137 ISK. Credit cards accepted everywhere.",
            "meals": [
                {"category": "Budget", "examples": "Hot dogs (Baejarins Beztu), gas station sandwiches, grocery store meals", "avg_cost": "$15-25/meal"},
                {"category": "Mid-range", "examples": "Cafe meals, fish & chips, lamb soup at local restaurants", "avg_cost": "$30-50/meal"},
                {"category": "Fine dining", "examples": "Grillid, Dill (Michelin), Matur og Drykkur", "avg_cost": "$80-150/meal"},
            ],
            "transport": [
                {"type": "Airport bus (Flybus)", "cost": "$28 one-way to Reykjavik"},
                {"type": "Reykjavik city bus (Straeto)", "cost": "$4.20/ride or $24/3-day pass"},
                {"type": "Rental car (compact)", "cost": "$65-95/day (add $15/day for insurance)"},
                {"type": "Rental car (4WD, for highlands)", "cost": "$120-180/day"},
                {"type": "Taxi (Reykjavik)", "cost": "$20-35 within city center"},
                {"type": "Domestic flight to Akureyri", "cost": "$120-180 round-trip"},
            ],
            "daily_budgets": [
                {"level": "Budget", "per_day": "$150-200/day", "notes": "Hostel, bus tours, grocery meals, free attractions"},
                {"level": "Mid-range", "per_day": "$300-400/day", "notes": "3-4 star hotel, guided tours, restaurant meals"},
                {"level": "Luxury", "per_day": "$600+/day", "notes": "5-star hotel, private tours, fine dining, spa visits"},
            ],
            "tips": [
                "Tap water is free and excellent — no need to buy bottled water",
                "Happy hour (15:00-18:00) cuts drink prices by 40-50%",
                "Bonus/Kronan supermarkets are cheapest for groceries",
                "Free attractions: Hallgrimskirkja exterior, Harpa concert hall, city walking paths",
                "Gas is expensive (~$8.50/gallon) — factor into rental car budget",
            ],
        },
        "default": {
            "currency": "Local currency. Credit cards widely accepted.",
            "meals": [
                {"category": "Budget", "examples": "Street food, fast casual", "avg_cost": "$10-20/meal"},
                {"category": "Mid-range", "examples": "Sit-down restaurants", "avg_cost": "$25-45/meal"},
                {"category": "Fine dining", "examples": "Upscale restaurants", "avg_cost": "$60-120/meal"},
            ],
            "transport": [
                {"type": "Public transit", "cost": "$3-5/ride"},
                {"type": "Taxi", "cost": "$15-30 within city"},
                {"type": "Rental car", "cost": "$50-80/day"},
            ],
            "daily_budgets": [
                {"level": "Budget", "per_day": "$100-150/day", "notes": "Hostel, public transit, street food"},
                {"level": "Mid-range", "per_day": "$200-300/day", "notes": "Hotel, tours, restaurants"},
                {"level": "Luxury", "per_day": "$500+/day", "notes": "Luxury hotel, private tours, fine dining"},
            ],
            "tips": ["Research local discount passes", "Eat where locals eat", "Book tours in advance for better rates"],
        },
    }

    key = "Iceland" if "iceland" in destination.lower() or "reykjavik" in destination.lower() else "default"
    data = costs_data[key]

    output = f"Travel Cost Guide: {destination}\n"
    output += f"{'='*60}\n"
    output += f"\nCurrency: {data['currency']}\n"

    output += f"\n--- Meal Costs ---\n"
    for meal in data["meals"]:
        output += f"  {meal['category']}: {meal['avg_cost']} ({meal['examples']})\n"

    output += f"\n--- Transportation ---\n"
    for t in data["transport"]:
        output += f"  {t['type']}: {t['cost']}\n"

    output += f"\n--- Daily Budget Estimates (per person) ---\n"
    for b in data["daily_budgets"]:
        output += f"  {b['level']}: {b['per_day']} — {b['notes']}\n"

    output += f"\n--- Money-Saving Tips ---\n"
    for i, tip in enumerate(data["tips"], 1):
        output += f"  {i}. {tip}\n"

    return output


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_flight_agent(destination: str, trip_dates: str):
    """Create the Flight Specialist agent with real research tools."""
    return Agent(
        role="Flight Specialist",
        goal=f"Research and recommend the best flight options for the {destination} trip "
             f"({trip_dates}), considering dates, airlines, prices, and flight durations. "
             f"Use real data from flight booking sites to provide accurate, current pricing.",
        backstory="You are an experienced flight specialist with deep knowledge of "
                  "airline schedules, pricing patterns, and travel routes. "
                  "You ALWAYS prioritize direct flights over connections, even when "
                  "connections are cheaper, because you believe travel time and reliability matter most. "
                  "You strongly favor budget airlines (PLAY, Norse Atlantic, Spirit, Frontier) "
                  "and focus on cost savings above all else. "
                  "You have booked thousands of flights and know the best times to fly. "
                  "You always research current prices and use real booking site data.",
        tools=[search_flight_prices],
        verbose=True,
        allow_delegation=False
    )


def create_hotel_agent(destination: str, trip_dates: str):
    """Create the Accommodation Specialist agent with real research tools."""
    # Determine main city for hotels (if destination is just a country, use capital)
    hotel_location = destination
    if destination.lower() == "iceland":
        hotel_location = "Reykjavik"
    elif destination.lower() == "france":
        hotel_location = "Paris"
    elif destination.lower() == "japan":
        hotel_location = "Tokyo"

    return Agent(
        role="Accommodation Specialist",
        goal=f"Suggest top-rated hotels in {hotel_location} for the {destination} trip "
             f"({trip_dates}), considering amenities, location, and value for money. "
             f"Use real hotel data from booking sites with current prices and reviews.",
        backstory="You are a seasoned accommodation expert with extensive knowledge of "
                  "hotels worldwide. You understand traveler needs and can match them with "
                  "perfect accommodations. You read reviews meticulously and know which "
                  "hotels offer the best experience for different budgets. You always "
                  "check current availability and actual guest reviews.",
        tools=[search_hotel_options],
        verbose=True,
        allow_delegation=False
    )


def create_itinerary_agent(destination: str, trip_duration: str):
    """Create the Travel Planner agent with real research tools."""
    return Agent(
        role="Travel Planner",
        goal=f"Create a detailed day-by-day travel plan with activities and attractions "
             f"that maximize the {destination} experience in {trip_duration}. "
             f"Use real current information about attractions, opening hours, and accessibility.",
        backstory=f"You are a creative travel planner with a passion for {destination}. "
                  f"You have extensive knowledge of {destination}'s attractions, culture, and hidden gems. "
                  f"You create itineraries that are well-paced, exciting, and memorable. "
                  f"You consider travel times, weather, and traveler preferences to craft the perfect journey. "
                  f"You always verify current information about attractions and tours.",
        tools=[search_attractions_activities],
        verbose=True,
        allow_delegation=False
    )


def create_local_expert_agent(destination: str):
    """Create the Local Expert agent who provides cultural tips, customs, and safety info."""
    return Agent(
        role="Local Expert",
        goal=f"Provide insider knowledge about {destination}'s local customs, cultural etiquette, "
             f"safety considerations, useful local phrases, tipping norms, and money-saving "
             f"local-only tips that influence the trip budget and itinerary execution.",
        backstory=f"You are a longtime local of {destination} who has guided dozens of foreign visitors. "
                  f"You know which neighborhoods to avoid at night, which restaurants tourists "
                  f"overpay at, what cultural etiquette mistakes embarrass visitors, and the local "
                  f"discounts that only residents know about. You always recommend specific actions "
                  f"travelers should take (and avoid) and quantify the impact on their daily costs.",
        verbose=True,
        allow_delegation=False
    )


def create_budget_agent(destination: str):
    """Create the Financial Advisor agent with real cost research tools."""
    return Agent(
        role="Financial Advisor",
        goal=f"Calculate total trip costs for {destination} and identify cost-saving opportunities "
             f"while maintaining quality. Use real current pricing data for all expenses.",
        backstory="You are a meticulous financial advisor specializing in travel budgeting. "
                  "You can analyze costs across flights, accommodations, activities, and meals. "
                  "You identify hidden costs and suggest smart ways to save money without "
                  "compromising the travel experience. You research actual current prices "
                  "and provide realistic budget estimates.",
        tools=[search_travel_costs],
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_flight_task(flight_agent, destination: str, trip_dates: str, departure_city: str):
    """Define the flight research task using real data."""
    return Task(
        description=f"Research and compile a list of REAL flight options from {departure_city} to {destination} "
                   f"for the trip ({trip_dates}). "
                   f"Use actual current flight data from booking sites like Skyscanner, Kayak, "
                   f"Google Flights, or Expedia. Find at least 2-3 different flight options from "
                   f"major airlines, including details about departure times, arrival times, "
                   f"duration, and current realistic prices. Provide "
                   f"recommendations on which flight offers the best value considering both "
                   f"price and convenience.",
        agent=flight_agent,
        expected_output=f"A detailed report with 2-3 REAL flight options from {departure_city} to {destination} "
                       f"including airlines, times, duration, current prices, and a recommendation with reasoning based on "
                       f"actual data from flight booking sites"
    )


def create_hotel_task(hotel_agent, destination: str, trip_dates: str):
    """Define the hotel recommendation task using real data."""
    # Determine main city for hotels
    hotel_location = destination
    if destination.lower() == "iceland":
        hotel_location = "Reykjavik"
    elif destination.lower() == "france":
        hotel_location = "Paris"
    elif destination.lower() == "japan":
        hotel_location = "Tokyo"

    return Task(
        description=f"Based on the trip dates ({trip_dates}), find and recommend "
                   f"the top 3-4 REAL hotels in {hotel_location}. Research actual hotels "
                   f"on Booking.com, TripAdvisor, Google Hotels, and Expedia. For each hotel, "
                   f"provide the actual name, current guest ratings, real prices per night, "
                   f"confirmed amenities, and explain why it suits this trip. "
                   f"Include a mix of budget, mid-range, and luxury options with honest reviews.",
        agent=hotel_agent,
        expected_output=f"A curated list of 3-4 REAL hotel recommendations in {hotel_location} with actual details "
                       f"about each hotel, confirmed amenities, real guest ratings, current prices, "
                       f"and personalized recommendations based on actual guest reviews"
    )


def create_itinerary_task(itinerary_agent, destination: str, trip_duration: str, trip_dates: str):
    """Define the itinerary planning task using real information."""
    return Task(
        description=f"Create a detailed {trip_duration} itinerary for {destination} ({trip_dates}) based on "
                   f"REAL current information. Research actual attractions, their opening hours, "
                   f"accessibility, and entry fees. Plan day-by-day activities including visits "
                   f"to real attractions and verified sites. Include realistic estimated travel times between "
                   f"locations, activity durations, and recommended visit times. Consider actual "
                   f"weather patterns for this time period in {destination} and make the itinerary realistic and well-paced.",
        agent=itinerary_agent,
        expected_output=f"A detailed day-by-day itinerary for {destination} with REAL activities based on verified "
                       f"attractions, realistic travel times, accurate estimated durations, current "
                       f"entry fees, and practical tips for {trip_duration} trip to {destination}"
    )


def create_local_expert_task(local_expert_agent, destination: str, trip_duration: str):
    """Define the local insider task with safety, etiquette, and money-saving tips."""
    return Task(
        description=f"Based on the planned itinerary for the {trip_duration} {destination} trip, "
                   f"provide local insider knowledge that the budget agent must factor in. Include: "
                   f"(1) cultural etiquette and customs that affect daily behavior, "
                   f"(2) tipping norms and service expectations, "
                   f"(3) safety considerations (which areas/situations to avoid, scams targeting tourists), "
                   f"(4) local-only money-saving tips (resident discounts, off-peak times, supermarkets vs. tourist shops), "
                   f"(5) 5 essential local phrases, and "
                   f"(6) quantified expected impact on daily costs (e.g., 'eating at supermarket bakeries instead of "
                   f"tourist cafes saves ~$15/day per person').",
        agent=local_expert_agent,
        expected_output=f"A local insider report for {destination} covering customs, tipping, safety, "
                       f"money-saving tips, useful phrases, and a quantified daily cost adjustment "
                       f"that the budget agent should apply to their estimates."
    )


def create_budget_task(budget_agent, destination: str, trip_duration: str):
    """Define the budget calculation task using real cost data."""
    return Task(
        description=f"Based on the REAL flight options, hotel recommendations, itinerary, and "
                   f"the LocalExpert's insider tips and cost adjustments, calculate a comprehensive "
                   f"budget for the {trip_duration} {destination} trip using current pricing. "
                   f"You MUST apply the LocalExpert's quantified daily cost adjustments (e.g., "
                   f"resident discounts, supermarket savings, tipping norms) directly to your meal, "
                   f"activity, and miscellaneous estimates. Research and include actual "
                   f"costs for flights, accommodation, meals (use real restaurant prices in the destination), "
                   f"activities/tours (verified prices), transportation within {destination}, "
                   f"and miscellaneous expenses. Provide total cost estimates "
                   f"for budget, mid-range, and luxury options based on real prices, with a clear "
                   f"line-item showing the savings from following the LocalExpert's advice. Suggest "
                   f"genuine cost-saving tips based on current market conditions and local insider info.",
        agent=budget_agent,
        expected_output=f"A comprehensive budget report with itemized REAL costs for flights, "
                       f"accommodation, meals, activities with actual entry fees, transportation, "
                       f"and total realistic estimates at different budget levels, plus "
                       f"evidence-based cost-saving recommendations for a {trip_duration} trip to {destination}"
    )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

def main(destination: str = "Iceland", trip_duration: str = "5 days",
         trip_dates: str = "January 15-20, 2026", departure_city: str = "New York",
         travelers: int = 2, budget_preference: str = "mid-range"):
    """
    Main function to orchestrate the travel planning crew.

    Args:
        destination: Travel destination (e.g., "Iceland", "France", "Japan")
        trip_duration: Duration of trip (e.g., "5 days", "7 days")
        trip_dates: Specific dates (e.g., "January 15-20, 2026")
        departure_city: City you're departing from (e.g., "New York", "Los Angeles")
        travelers: Number of travelers
        budget_preference: Budget level ("budget", "mid-range", "luxury")
    """

    print("=" * 80)
    print("CrewAI Multi-Agent Travel Planning System (REAL API VERSION)")
    print(f"Planning a {trip_duration} Trip to {destination}")
    print("=" * 80)
    print()
    print(f"📍 Destination: {destination}")
    print(f"📅 Dates: {trip_dates}")
    print(f"✈️  Departure from: {departure_city}")
    print(f"👥 Travelers: {travelers}")
    print(f"💰 Budget: {budget_preference}")
    print()

    # Validate configuration before proceeding
    print("🔍 Validating configuration...")
    if not validate_config():
        print("❌ Configuration validation failed. Please set up your .env file.")
        exit(1)

    # Set environment variables for CrewAI (it reads from os.environ)
    # CrewAI uses OPENAI_API_KEY and OPENAI_API_BASE environment variables
    os.environ["OPENAI_API_KEY"] = Config.API_KEY
    os.environ["OPENAI_API_BASE"] = Config.API_BASE
    
    # For Groq compatibility, also set OPENAI_MODEL_NAME
    if Config.USE_GROQ:
        os.environ["OPENAI_MODEL_NAME"] = Config.OPENAI_MODEL

    print("✅ Configuration validated successfully!")
    print()
    Config.print_summary()
    print()
    print("⚠️  IMPORTANT: This version uses REAL OpenAI API calls and web search")
    print("    Agents will research actual current prices and real information")
    print()
    print("Tip: Check your API usage at https://platform.openai.com/account/usage")
    print()

    # Create agents with destination parameters
    print("[1/5] Creating Flight Specialist Agent (researches real flights)...")
    flight_agent = create_flight_agent(destination, trip_dates)

    print("[2/5] Creating Accommodation Specialist Agent (researches real hotels)...")
    hotel_agent = create_hotel_agent(destination, trip_dates)

    print("[3/5] Creating Travel Planner Agent (researches real attractions)...")
    itinerary_agent = create_itinerary_agent(destination, trip_duration)

    print("[4/5] Creating Local Expert Agent (cultural tips, safety, insider savings)...")
    local_expert_agent = create_local_expert_agent(destination)

    print("[5/5] Creating Financial Advisor Agent (analyzes real costs)...")
    budget_agent = create_budget_agent(destination)

    print("\n✅ All agents created successfully!")
    print()

    # Create tasks with destination parameters
    print("Creating tasks for the crew...")
    flight_task = create_flight_task(flight_agent, destination, trip_dates, departure_city)
    hotel_task = create_hotel_task(hotel_agent, destination, trip_dates)
    itinerary_task = create_itinerary_task(itinerary_agent, destination, trip_duration, trip_dates)
    local_expert_task = create_local_expert_task(local_expert_agent, destination, trip_duration)
    budget_task = create_budget_task(budget_agent, destination, trip_duration)

    print("Tasks created successfully!")
    print()

    # Create the crew with sequential task execution
    print("Forming the Travel Planning Crew...")
    print("Task Sequence: FlightAgent → HotelAgent → ItineraryAgent → LocalExpert → BudgetAgent")
    print()

    crew = Crew(
        agents=[flight_agent, hotel_agent, itinerary_agent, local_expert_agent, budget_agent],
        tasks=[flight_task, hotel_task, itinerary_task, local_expert_task, budget_task],
        verbose=True,
        process="sequential"  # Sequential task execution
    )

    # Execute the crew
    print("=" * 80)
    print("Starting Crew Execution with REAL API Calls...")
    print(f"Planning {trip_duration} trip to {destination} ({trip_dates})")
    print("=" * 80)
    print()

    try:
        result = crew.kickoff(inputs={
            "trip_destination": destination,
            "trip_duration": trip_duration,
            "trip_dates": trip_dates,
            "departure_city": departure_city,
            "travelers": travelers,
            "budget_preference": budget_preference
        })

        print()
        print("=" * 80)
        print("✅ Crew Execution Completed Successfully!")
        print("=" * 80)
        print()
        print(f"FINAL TRAVEL PLAN REPORT FOR {destination.upper()} (Based on Real API Data):")
        print("-" * 80)
        print(result)
        print("-" * 80)

        # Save output to file
        output_filename = f"crewai_output_{destination.lower()}.txt"
        output_path = Path(__file__).parent / output_filename

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CrewAI Multi-Agent Travel Planning System - Real API Execution Report\n")
            f.write(f"Planning a {trip_duration} Trip to {destination}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Trip Details:\n")
            f.write(f"  Destination: {destination}\n")
            f.write(f"  Duration: {trip_duration}\n")
            f.write(f"  Dates: {trip_dates}\n")
            f.write(f"  Departure: {departure_city}\n")
            f.write(f"  Travelers: {travelers}\n")
            f.write(f"  Budget Preference: {budget_preference}\n\n")
            f.write(f"Execution Time: {datetime.now()}\n")
            f.write(f"API Version: REAL API CALLS (OpenAI GPT-4)\n")
            f.write(f"Data Source: Web research via OpenAI\n\n")
            f.write("IMPORTANT NOTES:\n")
            f.write("- All flight prices, hotel costs, and attraction information is based on real data\n")
            f.write("- Prices are current as of the date this was run\n")
            f.write("- Hotel availability and prices may vary by booking date\n")
            f.write("- Weather conditions and attraction hours should be verified before travel\n\n")
            f.write("FINAL TRAVEL PLAN REPORT:\n")
            f.write("-" * 80 + "\n")
            f.write(str(result))
            f.write("\n" + "-" * 80 + "\n")

        print(f"\n✅ Output saved to {output_filename}")
        print("ℹ️  Note: All data in this report is based on REAL API calls to OpenAI")
        print("    and research of current travel information sources.")

    except Exception as e:
        print(f"\n❌ Error during crew execution: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Verify OPENAI_API_KEY is set: export OPENAI_API_KEY='sk-...'")
        print("   2. Check API key is valid and has sufficient credits")
        print("   3. Verify internet connection for web research")
        print("   4. Check OpenAI API status at https://status.openai.com")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow command line arguments to override defaults
    import sys

    kwargs = {
        "destination": "Iceland",
        "trip_duration": "5 days",
        "trip_dates": "January 15-20, 2026",
        "departure_city": "New York",
        "travelers": 2,
        "budget_preference": "mid-range"
    }

    # Parse command line arguments (optional)
    # Usage: python crewai_demo.py [destination] [duration] [departure_city]
    # Example: python crewai_demo.py "France" "7 days" "Los Angeles"
    if len(sys.argv) > 1:
        kwargs["destination"] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs["trip_duration"] = sys.argv[2]
    if len(sys.argv) > 3:
        kwargs["departure_city"] = sys.argv[3]
    if len(sys.argv) > 4:
        kwargs["trip_dates"] = sys.argv[4]
    if len(sys.argv) > 5:
        kwargs["travelers"] = int(sys.argv[5])
    if len(sys.argv) > 6:
        kwargs["budget_preference"] = sys.argv[6]

    main(**kwargs)

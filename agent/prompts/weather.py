# ==========================================================================
# Master Thesis - Weather Agent Prompt
#   - André Filipe Gomes Silvestre, 20240502
# 
#   Focused prompt for the weather specialist agent.
#   Ultra-concise for fast inference on small models.
# ==========================================================================

from datetime import datetime

WEATHER_AGENT_PROMPT = """You are a **Weather Specialist** for Lisbon. Use ONLY IPMA tools - NEVER invent data.

# TOOLS
- `get_current_weather_summary`: Today's weather
- `get_weather_forecast`: Up to 5 days forecast (MAX 5 days!)
- `get_weather_warnings`: Active weather alerts

# ⚠️ LOCATION LIMITATION ⚠️
Weather data is ONLY available for **Lisboa city** (IPMA station).
If user asks about Sintra, Cascais, Setúbal, or other nearby areas, explain:
"Só tenho dados meteorológicos para Lisboa. [Local] costuma ter clima semelhante, 
embora possa ser ligeiramente mais fresco/chuvoso devido à proximidade das serras/costa.
Aqui está a previsão de Lisboa como referência..."

# RULES
1. **PT-PT ONLY**: Use "está sol", "vai chover", NEVER Brazilian Portuguese
2. **5-DAY LIMIT**: If asked beyond 5 days, say "Só tenho previsões até 5 dias"
3. **ALWAYS call tools** - never guess weather data
4. **Include practical tips**: Umbrella, sunscreen, jacket recommendations
5. **Use emojis**: ☀️🌤️🌧️⛈️🌡️💨
6. **CORRECT DAY NAMES**: Today is {current_date}. Count forward correctly!

# OUTPUT FORMAT
After getting tool results, respond naturally with:
- Current conditions (temperature, sky)
- Precipitation probability
- Warnings if any
- Practical advice

Date: {current_date} | Time: {current_time}
"""


def get_weather_prompt() -> str:
    """Returns weather agent prompt with current date/time."""
    now = datetime.now()
    return WEATHER_AGENT_PROMPT.format(
        current_date=now.strftime("%A, %B %d, %Y"),
        current_time=now.strftime("%H:%M")
    )


# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    print("\033[1m" + "=" * 60 + "\033[0m")
    print("\033[1m🧪 Weather Agent Prompt Test\033[0m")
    print("\033[1m" + "=" * 60 + "\033[0m")
    
    prompt = get_weather_prompt()
    print(f"\n\033[1m📝 Prompt Preview:\033[0m")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print(f"\n\033[1mTotal length:\033[0m {len(prompt)} characters (~{len(prompt)//4} tokens)")
    print(f"\033[1;32m✅ Weather prompt loaded!\033[0m")

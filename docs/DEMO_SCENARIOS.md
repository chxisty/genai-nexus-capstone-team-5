# Interactive Demo Scenarios: Railway Passenger Help Desk

**GenAI Nexus Fellowship · Module 2 Capstone**  
**Team 5 (Neuroverse)**  
**Integration Lead:** Ram Narisetti (Member 5)

---

## 1. Overview

This document provides 10 structured demonstration scenarios for evaluating the Railway Passenger Help Desk terminal application (`main.py`). Each scenario lists the terminal input, tool invocation flow, expected assistant output, and the underlying architectural guarantee verified.

To start the demo, launch the terminal REPL:
```bash
python main.py
```

---

## Scenario 1: Greeting, Banner & In-App Help

- **User Input:**
  ```text
  /help
  ```
- **Execution Flow:** Handled directly by REPL command parser without invoking LLM.
- **Expected Output:**
  ```text
  Available Commands:
    /help    - Show this help message
    /new     - Start a new conversation (fresh session)
    /history - Show history of current session
    /exit    - Exit the application
  ```
- **Feature Demonstrated:** Fast, deterministic command dispatch.

---

## Scenario 2: Station Search

- **User Input:**
  ```text
  You: Search for stations in Delhi
  ```
- **Execution Flow:**
  1. LLM identifies station lookup intent.
  2. Dispatches `station_search(query="Delhi")`.
  3. `ToolRegistry` dispatches to `tools.database_tools.search_stations`.
  4. Database queries SQLite `stations` table for `%Delhi%`. Matches `NDLS`, `DLI`, `NZM`, `DEC`, etc.
  5. Grounded synthesis formats passenger answer.
- **Expected Output:**
  ```text
  Assistant:
  Here are the stations matching Delhi in the railway timetable:
  - New Delhi (NDLS)
  - Delhi Junction (DLI)
  - Hazrat Nizamuddin (NZM)
  - Delhi Cantt (DEC)
  - Delhi Sarai Rohilla (DEE)

  Sources:
  None (No official document cited)

  SQL:
  null
  ```
- **Feature Demonstrated:** Database lookup via `station_search` tool.

---

## Scenario 3: Train Availability Between Stations

- **User Input:**
  ```text
  You: Which trains are available from LTT to GWL?
  ```
- **Execution Flow:**
  1. LLM emits tool call `train_search(source="LTT", destination="GWL", date="2026-11-15")`.
  2. `tools.database_tools.search_trains` resolves station codes `LTT` and `GWL`, queries joined `schedules` and `trains` tables.
  3. Finds 6 scheduled trains.
  4. Assistant returns verified train numbers and names.
- **Expected Output:**
  ```text
  Assistant:
  Based on the timetable database, 6 trains operate between Lokmanya Tilak Terminus (LTT) and Gwalior (GWL):
  - 01101 (Mumbai LTT - Gwalior Special)
  - 01127 (Mumbai LTT - Gwalior Special)
  - 11079 (Mumbai LTT - Gorakhpur Express)
  - 12137 (Punjab Mail)
  - 12171 (Mumbai LTT - Haridwar AC Superfast Express)
  - 04116 (Mumbai LTT - Subedarganj Special)

  Sources:
  None (No official document cited)

  SQL:
  null
  ```
- **Feature Demonstrated:** Real multi-table timetable query matching source and destination stops.

---

## Scenario 4: Train Details & Available Classes

- **User Input:**
  ```text
  You: What are the details and classes for train 01101?
  ```
- **Execution Flow:**
  1. Dispatches `train_details(train_number="01101")`.
  2. Queries SQLite `trains` joined with `train_types`.
  3. Returns classes `['3A', 'SL']`, type `Special`, origin `LTT`, destination `GWL`.
- **Expected Output:**
  ```text
  Assistant:
  Train 01101 is the Mumbai LTT - Gwalior (Weekly) Special.
  - Origin: Lokmanya Tilak Terminus (LTT)
  - Destination: Gwalior (GWL)
  - Train Type: Special
  - Available Classes: 3AC (3A) and Sleeper (SL)

  Note: Timetable data is static and does not reflect current seat availability.

  Sources:
  None (No official document cited)

  SQL:
  null
  ```
- **Feature Demonstrated:** Class decoding from boolean flags (`has_3a`, `has_sl`) and static disclaimers.

---

## Scenario 5: Route Lookup & Sequential Halts

- **User Input:**
  ```text
  You: What is the route for train 01101?
  ```
- **Execution Flow:**
  1. Dispatches `route_lookup(train_number="01101")`.
  2. Retrieves 152 schedule records ordered by `stop_sequence`.
  3. Assistant summarizes major commercial halts.
- **Expected Output:**
  ```text
  Assistant:
  The route for train 01101 (Mumbai LTT - Gwalior Special) includes stops at:
  Lokmanya Tilak Terminus (departure 15:50), Thane, Kalyan, Igatpuri, Nasik Road, Manmad, Jalgaon, Bhusaval, Khandwa, Itarsi, Bhopal, Bina, Jhansi, and terminates at Gwalior (arrival 13:40).

  Sources:
  None (No official document cited)

  SQL:
  null
  ```
- **Feature Demonstrated:** Stop sequence aggregation without passing phantom non-stopping tracks.

---

## Scenario 6: Multi-Turn Context & Pronoun Follow-Up

- **User Input (Turn 1):**
  ```text
  You: Tell me about train 12002
  ```
- **Assistant Response (Turn 1):**
  ```text
  Assistant:
  Train 12002 is the New Delhi - Habibganj Shatabdi Express operating from New Delhi (NDLS) to Habibganj (HBJ). Available classes: Executive Chair Car (EC) and AC Chair Car (CC).
  ```
- **User Input (Turn 2):**
  ```text
  You: What time does it leave New Delhi?
  ```
- **Execution Flow (Turn 2):**
  1. `ContextHandler` detects pronoun `"it"` referring to train 12002.
  2. Memory adapter feeds conversation history into LLM prompt.
  3. Assistant identifies departure time (06:15) from verified schedule data.
- **Expected Output (Turn 2):**
  ```text
  Assistant:
  Train 12002 (Bhopal Shatabdi) departs from New Delhi (NDLS) at 06:15.

  Sources:
  None (No official document cited)

  SQL:
  null
  ```
- **Feature Demonstrated:** Cross-turn conversational memory and pronoun resolution.

---

## Scenario 7: Unsupported Live Running Status

- **User Input:**
  ```text
  You: What is the live running status of train 12951 right now?
  ```
- **Execution Flow:**
  1. LLM / System grounding policy identifies live-status inquiry.
  2. The 2016 static database has no real-time telemetry.
  3. System declines with honest disclaimer.
- **Expected Output:**
  ```text
  Assistant:
  I don't know. The system operates on a static railway timetable snapshot and does not have access to live train running status, GPS tracking, platform numbers, or current delays.

  Sources:
  None (No official document cited)

  SQL:
  null
  ```
- **Feature Demonstrated:** Zero hallucination on dynamic data; `dont_know: true`.

---

## Scenario 8: Railway Policy Query (Document RAG Absent)

- **User Input:**
  ```text
  You: What is the cancellation charge for a confirmed ticket cancelled 48 hours before departure?
  ```
- **Execution Flow:**
  1. Assistant checks evidence accumulator: no document RAG module is present in the repository.
  2. System detects absence of verified document chunks.
  3. `_finalize_response` returns honest refusal without inventing cancellation slabs.
- **Expected Output:**
  ```text
  Assistant:
  I don't know. I could not find supporting evidence in the official railway documents or timetable database to answer that.

  Sources:
  None (No official document cited)

  SQL:
  null
  ```
- **Feature Demonstrated:** Strict refusal when required knowledge base is absent; zero fabricated rules.

---

## Scenario 9: Handling Upstream API Failures Gracefully

- **Scenario:** Network disconnect or invalid API key.
- **Simulated Test:** `test_missing_api_key_graceful_handling`.
- **Execution Flow:**
  1. LLM client raises `LLMAuthError` or `LLMConnectionError`.
  2. `AIAssistant` catches error, redacts any sensitive tokens.
  3. Formats fallback `GroundedAnswer` with polite disclaimer.
- **Expected Output:**
  ```text
  Assistant:
  I don't know. I encountered a temporary problem contacting the AI service. Please try again.

  Sources:
  None (No official document cited)

  SQL:
  null
  ```
- **Feature Demonstrated:** Error containment: terminal application does not crash or dump stack traces.

---

## Scenario 10: Session Persistence & Inspection (`/history`, `/new`)

- **User Input (Turn 1):**
  ```text
  You: Search for station code JP
  ```
- **Assistant Response:**
  ```text
  Assistant:
  Station code JP corresponds to Jaipur Junction in North Western Railway.
  ```
- **User Input (Turn 2):**
  ```text
  You: /history
  ```
- **Expected Output (Turn 2):**
  ```text
  --- Session History (sess_a319f02c) ---
  [User]: Search for station code JP
  [Assistant]: Station code JP corresponds to Jaipur Junction in North Western Railway.
  ----------------------------------------
  ```
- **User Input (Turn 3):**
  ```text
  You: /new
  ```
- **Expected Output (Turn 3):**
  ```text
  Started a new conversation session: sess_c8b417e9
  ```
- **User Input (Turn 4):**
  ```text
  You: /history
  ```
- **Expected Output (Turn 4):**
  ```text
  No history recorded for this session yet.
  ```
- **Feature Demonstrated:** Full SQLite session lifecycle management via `SessionManager`.

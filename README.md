# Seminar Hall Allocation AI Assistant

A complete Python college-assignment prototype based on the supplied PDF:

**Development of a Heuristic Function for Seminar Hall Allocation**

The project provides:

1. A Streamlit chatbot interface.
2. PDF-based question answering using TF-IDF retrieval.
3. A seminar hall allocation calculator.
4. The assignment's heuristic function:
   `H(A) = 2(U) + 20(M) + 100(Cl) + 1(Dist)`
5. A small Hill Climbing demonstration.
6. Sample hall data based on the halls shown in the assignment.

## 1. Requirements

- Python 3.10 or newer
- Internet is only needed once to install Python packages.
- No OpenAI API key is required for this version.

## 2. Installation

Open a terminal in this project folder:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 3. Run

```bash
streamlit run app.py
```

The browser should open the chatbot automatically. If not, open the local URL shown in the terminal.

## 4. Example questions

Try:

```text
What is the heuristic function?
```

```text
Explain Hill Climbing
```

```text
What are the objectives?
```

```text
What are the limitations?
```

For allocation:

```text
Allocate a hall for 120 participants with projector, mic and AC
```

You can also include a time range:

```text
Allocate a hall for 120 participants with projector, mic and AC from 10 AM to 12 PM
```

## 5. How the chatbot works

### PDF Question Answering

The application extracts text from the supplied PDF with `pypdf`.

The text is divided into chunks. TF-IDF converts the chunks and the user's question into vectors. Cosine similarity is then used to retrieve the most relevant chunks.

This is a lightweight retrieval-based chatbot and does not require a paid generative-AI API.

### Hall allocation

For every eligible hall:

- `U` = capacity - expected participants
- `M` = number of required facilities missing
- `Cl` = number of overlapping bookings
- `Dist` = distance in metres

The score is:

```text
H(A) = 2(U) + 20(M) + 100(Cl) + 1(Dist)
```

The lowest score is preferred because the assignment defines the problem as a minimization problem.

### Hill Climbing

The prototype also demonstrates the assignment's Hill Climbing idea:

1. Start with an allocation.
2. Calculate its heuristic.
3. Generate a neighboring allocation.
4. Compare the heuristic.
5. Accept a lower score.
6. Repeat until no better neighbor is found.

## 6. Project structure

```text
seminar_hall_chatbot/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── halls.json
│
└── knowledge/
    └── AI_Seminar_Hall_Allocation.pdf
```

## 7. Important academic note

The hall information in `data/halls.json` is the sample data from the supplied assignment. For a real college deployment, replace it with actual hall capacities, facilities, bookings and distances.

The current prototype intentionally follows the assignment's stated heuristic rather than silently changing its weights or criteria.

## 8. Viva explanation

**What is the purpose of the chatbot?**

It allows users to ask questions about the seminar-hall allocation assignment and also demonstrates how the heuristic can be used to select a suitable hall.

**Why use TF-IDF?**

It provides a simple local retrieval mechanism for matching a question to relevant PDF content without requiring a paid API.

**Why is the problem a minimization problem?**

The assignment defines a lower heuristic value as a more suitable allocation.

**Why is the clash weight high?**

The assignment gives booking clashes the largest penalty among its example weights because two events cannot use the same hall at the same time.

**What is the limitation of Hill Climbing?**

It can stop at a local optimum even when a better allocation exists through a sequence of changes.

## 9. Possible future enhancements

- Real database for halls and bookings
- Login for students/faculty
- Admin dashboard
- Calendar-based booking
- Conflict prevention
- Event priority
- Accessibility requirements
- Maintenance/cleaning periods
- Advanced search such as A* or simulated annealing
- LLM-based natural-language responses

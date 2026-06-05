# CodeTutor AI Frontend

Frontend for the AI-Based Personalised Programming Tutor final year project.

## Current Flow

- Landing page: `/`
- Authentication mock page: `/auth`
- Main dashboard: `/dashboard`
- AI tutor workspace: `/analyze`
- Lesson learning: `/lessons`
- Progress chart: `/progress`
- Badges and achievements: `/badges`
- Profile: `/profile`
- Settings: `/settings`

## Mock-First API Strategy

The frontend currently uses mock analysis data by default. This keeps the user flow working before
the FastAPI backend is wired.

To connect to the backend later:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_USE_MOCK_API=false
```

Then the tutor workspace will call:

```text
POST /analyze
```

with this payload:

```json
{
  "code": "print('hello')",
  "role": "student",
  "language": "python"
}
```

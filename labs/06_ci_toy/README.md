# Toy CI repository — Task 6.6 (red, then green)

This folder is a **throwaway repository** used for the CI practice exercise. It is
intentionally separate from VisionGuard so that breaking a test cannot affect the
main project.

## What it contains

- `.github/workflows/ci.yml` — runs on every push, matrix over **Python 3.10 and 3.11**,
  runs `pytest test_toy.py`.
- `test_toy.py` — two tiny tests (`add` and `slugify`).

## Red → green procedure

1. Create an empty GitHub repository named `23L-0578_ci_toy` (throwaway).
2. From *this folder*:

   ```bash
   git init
   git branch -M main
   git remote add origin https://github.com/<user>/23L-0578_ci_toy.git
   git add .
   git commit -m "ci toy: initial green workflow"
   ```
3. **Break it:** edit `add()` to `return a - b`, commit and push → both matrix jobs
   fail (**red run**), capture the screenshot.
4. **Fix it:** restore `return a + b`, push → both matrix jobs pass (**green run**),
   capture the screenshot.

The screenshots belong in the report at §6.6 Toy Part.

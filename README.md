# Job-Lint

**Ever send 500 applications and get 50 rejection emails—5 of which actually had your name in them?**

A rule-based NLP tool for cutting through the nonsense in job postings. It catches inflated requirements, mismatched salaries, and corporate fluff—so you can focus on real opportunities instead of wasting time.

## Features

- **Experience Level Detection**: Parses years of experience requirements from job descriptions
- **Technology Stack Analysis**: Identifies programming languages and frameworks, distinguishing between required vs preferred skills
- **Salary Extraction & Normalization**: Extracts salary ranges and normalizes them to San Francisco cost of living for fair comparison
- **Location & Remote Work Detection**: Identifies job location and work arrangement (remote, hybrid, on-site)
- **Sophistication Scoring**: Calculates job complexity based on keywords related to architecture, mentoring, and technical leadership
- **Tier vs Level Classification**: Distinguishes between what companies claim to want (tier) vs actual job complexity (level)
- **Weighted Complexity Analysis**: Combines language breadth with role sophistication for realistic difficulty assessment

## Installation

```bash
# Clone the repository
git clone https://github.com/HimayGora/job-lint
cd job-lint

# Install dependencies
pip install -r requirements.txt

# Run on a job posting
python job_lint.py job-posting.txt
```

## Usage

### Basic Usage
```bash
# Analyze a single job posting
python job_lint.py path/to/job-posting.txt
```

### Example Output
```
Parsed Job Post:
Title: Senior Full-Stack Developer - FinTech Startup
Years of Experience: 5
Programming Languages (Score: 3.0):
  Javascript:
    Required: Core: Javascript, Typescript + Frameworks: Node.Js, React
    Preferred: Frameworks: Next.Js, Express
  Database:
    Required: Core: Postgresql
    Preferred: Core: Redis

Salary Information:
  Raw Range: $120,000 - $160,000
  SF Normalized: $230,769 - $307,692
  COL Multiplier: 0.52 (based on houston, tx)

Level Classification:
Recommended Tier: Middle
Recommended Level: Advanced
Weighted Complexity: 2.60
Overall Assessment: Middle Advanced (Complexity: 2.60)
```

## How It Works

### Language Family System
Technologies are organized into families (JavaScript, Python, Database, etc.) with core languages and frameworks scored differently:
- **Required skills**: 1.0 point per language family + 0.2 per framework
- **Preferred skills**: 0.1 point per language family + 0.02 per framework
- **Maximum score**: 6.0 (prevents keyword stuffing inflation)

### Sophistication Scoring
Jobs are scored based on complexity keywords:
- **High complexity** (3 points): architecture, mentoring, system design
- **Medium complexity** (2 points): ownership, scalability, leadership
- **Low complexity** (1 point): code reviews, training, best practices
- **Score range**: 10-50 points

### Tier vs Level Analysis
- **Tier**: Based on experience requirements and language count
- **Level**: Based on sophistication score (Entry → Boundary)
- **Weighted Complexity**: Language score × sophistication percentage

## Technology Coverage

Supports analysis of 12+ technology families:
- JavaScript/TypeScript ecosystem
- Python and frameworks
- Java/JVM languages
- Systems languages (C, C++, Rust, Go)
- .NET stack
- PHP frameworks
- Ruby/Rails
- Database technologies
- DevOps/Infrastructure tools
- Mobile development
- Web frontend technologies

## Development Status

job-lint is in active development (Sprint 3/16). The API and data structures may change significantly before v1.0.

### Roadmap
- **Sprint 4-8**: Red flag detection, corporate BS patterns, batch processing
- **Sprint 9-12**: Resume parsing, enhanced fuzzy matching, performance optimization  
- **Sprint 13-16**: CLI wrapper, install scripts, contributor guidelines

## Contributing

Contributions will be welcomed after the initial development phase is complete (Sprint 16). The project will accept:
- New regex patterns for experience/salary extraction
- Additional programming languages and frameworks
- Cost of living data for more cities
- Bug reports with sample job postings

## Use Cases

- **Job Seekers**: Filter out roles with unrealistic requirements before applying
- **Recruiters**: Validate job posting requirements against market standards
- **HR Teams**: Ensure job descriptions match actual role complexity
- **Salary Research**: Compare compensation across different markets

## Testing

*Coming in a soon in a sprint near you*

## Architecture

- **Rule-based NLP**: Uses regex patterns and keyword matching for reliability
- **Zero external dependencies**: Works offline without API calls
- **Modular design**: Separate extraction methods for each data type
- **Extensible patterns**: Easy to add new languages, locations, or scoring criteria

## License

AGPL-3.0


## Requirements
- Python 3.8+
- No external dependencies beyond standard library

---

Built for developers who want to make smarter job application decisions based on data, not marketing copy.

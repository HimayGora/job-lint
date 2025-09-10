import numpy as np
import re
from typing import Dict, List, Optional, Union, Tuple, Any


class JobPostParser:
	def __init__(self):
		self.experience_patterns = [
			r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|yrs?)',  # "4-6 years" or "4 to 6 years"
			r'(\d+)\+\s*(?:years?|yrs?)',  # "4+ years" (explicit plus)
			r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',  # "4 years of experience"
			r'minimum\s*(\d+)\s*(?:years?|yrs?)',
			r'at least\s*(\d+)\s*(?:years?|yrs?)',
		]
		
		# Language families and frameworks for weighted scoring
		self.language_families = {
			'javascript': {
				'core': ['javascript', 'js', 'typescript', 'ts'],
				'frameworks': ['node.js', 'nodejs', 'react', 'angular', 'vue', 'next.js', 'gatsby', 'express']
			},
			'web_frontend': {
				'core': ['html', 'css', 'sass', 'scss', 'less'],
				'frameworks': ['bootstrap', 'tailwind', 'material-ui', 'styled-components']
			},
			'python': {
				'core': ['python'],
				'frameworks': ['django', 'flask', 'fastapi', 'pandas', 'numpy', 'tensorflow', 'pytorch']
			},
			'java_jvm': {
				'core': ['java', 'scala', 'kotlin', 'groovy'],
				'frameworks': ['spring', 'spring boot', 'hibernate', 'maven', 'gradle']
			},
			'systems': {
				'core': ['c', 'c\\+\\+', 'rust', 'go', 'zig'],
				'frameworks': []
			},
			'dotnet': {
				'core': ['c#', '.net', 'f#', 'vb.net'],
				'frameworks': ['asp.net', 'entity framework', 'blazor']
			},
			'php': {
				'core': ['php'],
				'frameworks': ['laravel', 'symfony', 'codeigniter', 'wordpress']
			},
			'ruby': {
				'core': ['ruby'],
				'frameworks': ['rails', 'ruby on rails', 'sinatra']
			},
			'database': {
				'core': ['sql', 'postgresql', 'mysql', 'sqlite', 'mongodb', 'redis'],
				'frameworks': ['sequelize', 'prisma', 'mongoose', 'sqlalchemy']
			},
			'shell_devops': {
				'core': ['bash', 'shell', 'powershell', 'zsh'],
				'frameworks': ['docker', 'kubernetes', 'terraform', 'ansible']
			},
			'mobile': {
				'core': ['swift', 'objective-c', 'dart'],
				'frameworks': ['flutter', 'react native', 'xamarin', 'ionic']
			},
			'other': {
				'core': ['r', 'matlab', 'perl', 'lua', 'haskell', 'erlang', 'elixir'],
				'frameworks': []
			}
		}
		
		self.education_patterns = [
			r'bachelor[\s\']*s?\s*(?:degree|diploma)?',
			r'master[\s\']*s?\s*(?:degree|diploma)?',
			r'phd|ph\.d|doctorate',
			r'associate[\s\']*s?\s*(?:degree|diploma)?',
			r'b\.s\.|b\.a\.|m\.s\.|m\.a\.|m\.b\.a\.|mba',
			r'computer science|cs|software engineering|engineering',
			r'information technology|informatics',
			r'no experience required',
			r'entry level',
			r'willing to train',
			r'equivalent experience',
		]
		
		# NEW: Salary extraction patterns
		self.salary_patterns = [
			r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:to|-|–)\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s+year|annually|/year|yearly)?',  # $80,000 - $120,000
			r'\$(\d{1,3}(?:,\d{3})*(?:k|K))\s*(?:to|-|–)\s*\$(\d{1,3}(?:,\d{3})*(?:k|K))\s*(?:per\s+year|annually|/year|yearly)?',  # $80K - $120K
			r'(\d{1,3}(?:,\d{3})*)\s*(?:to|-|–)\s*(\d{1,3}(?:,\d{3})*)\s*(?:per\s+year|annually|/year|yearly|salary)',  # 80000 to 120000 per year
			r'salary:?\s*\$?(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*(?:to|-|–)\s*\$?(\d{1,3}(?:,\d{3})*(?:k|K)?)',  # Salary: $80K - $120K
			r'up to \$(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*(?:per\s+year|annually|/year|yearly)?',  # up to $120K
			r'starting at \$(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*(?:per\s+year|annually|/year|yearly)?',  # starting at $80K
			r'\$(\d{1,3}(?:,\d{3})*(?:k|K)?)\s*(?:per\s+year|annually|/year|yearly)',  # $120K per year
			r'(\d{1,3}(?:,\d{3})*(?:k|K))\s*(?:per\s+year|annually|/year|yearly)',  # 120K per year
		]
		
		# Keep original location patterns from working version
		self.location_patterns = [
			r'(?:location|based\s+in|office\s+in|headquarters|hq)\s*:?\s*([^,\n\.]+(?:,[^,\n\.]+)?)',  # location: City, ST
			r'(remote|hybrid|on-site|work\s+from\s+home|wfh|fully\s+remote)',
			r'([A-Z][a-zA-Z\s]+,\s*[A-Z]{2}(?:\s+\d{5})?)',  # City, ST 12345
			r'([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)',  # City, Country
			r'(\d+\s*%\s*remote|\d+%\s*remote|100%\s*remote)',
			r'(?:remote|hybrid)\s*(?:work|position|job)?(?:\s*-\s*([^,\n\.]+))?',  # remote - location
		]
		
		# NEW: Cost of living multipliers for salary normalization to SF
		self.col_multipliers = {
			# Major tech cities
			'san francisco': 1.0, 'sf': 1.0, 'san fran': 1.0,
			'new york': 0.95, 'nyc': 0.95, 'manhattan': 0.95,
			'seattle': 0.85, 'bellevue': 0.85,
			'los angeles': 0.75, 'la': 0.75, 'santa monica': 0.75,
			'boston': 0.80, 'cambridge': 0.80,
			'washington dc': 0.82, 'dc': 0.82, 'washington': 0.82,
			
			# Other expensive areas
			'palo alto': 1.05, 'mountain view': 1.0, 'sunnyvale': 0.95,
			'cupertino': 0.95, 'san jose': 0.90, 'santa clara': 0.90,
			'berkeley': 0.90, 'oakland': 0.85, 'san mateo': 0.95,
			
			# Mid-tier cities
			'denver': 0.70, 'boulder': 0.70,
			'austin': 0.68, 'portland': 0.70, 'chicago': 0.72,
			'san diego': 0.75, 'miami': 0.65, 'atlanta': 0.60,
			'raleigh': 0.58, 'nashville': 0.58, 'phoenix': 0.60,
			
			# Lower cost areas
			'dallas': 0.55, 'houston': 0.52, 'kansas city': 0.48,
			'tampa': 0.50, 'orlando': 0.50, 'cleveland': 0.45,
			'detroit': 0.45, 'columbus': 0.48, 'milwaukee': 0.48,
			'salt lake city': 0.55, 'minneapolis': 0.58, 'omaha': 0.45,
			
			# Remote/default
			'remote': 0.75, 'anywhere': 0.75, 'usa': 0.65,
		}
		
		# Classification criteria
		self.experience_levels = {
			'junior': {'min_years': 0, 'max_years': 2},
			'middle': {'min_years': 3, 'max_years': 5},
			'senior': {'min_years': 6, 'max_years': 100}
		}
		
		self.language_requirements = {
			'junior': 1,	# At least 1 language
			'middle': 2,	# At least 2 languages
			'senior': 3		# At least 3 languages
		}

		
		self.education_requirements = {
			'junior': ['associate', 'bachelor', 'willing to train', 'entry level', 'no experience required'],
			'middle': ['bachelor', "bachelor's preferred"],  
			'senior': ['bachelor', 'master', 'phd', 'equivalent experience'],
		}

	def pattern_match(self, text: str, patterns: List[str], 
					 return_type: str = 'first', 
					 flags: int = re.IGNORECASE) -> Union[List[str], str, None]:
		"""
		Base function for pattern matching with flexible return options.
		
		Args:
			text: Text to search in
			patterns: List of regex patterns to match against
			return_type: 'first' (first match), 'all' (all matches), 'exists' (bool)
			flags: Regex flags to use
			
		Returns:
			Matched results based on return_type
		"""
		all_matches = []
		
		for pattern in patterns:
			matches = re.findall(pattern, text, flags)
			if matches:
				if return_type == 'first':
					if isinstance(matches[0], tuple):
						return matches[0]
					return matches[0]
				elif return_type == 'exists':
					return True
				else:
					all_matches.extend(matches)
		
		if return_type == 'exists':
			return False
		elif return_type == 'all':
			return all_matches
		
		return None

	def extract_with_context(self, text: str, patterns: List[str], 
						   context_lines: int = 0) -> List[Tuple[str, str]]:
		"""
		Extract matches with surrounding context lines.
		
		Args:
			text: Text to search in
			patterns: List of regex patterns
			context_lines: Number of lines before/after to include
			
		Returns:
			List of (match, context) tuples
		"""
		lines = text.split('\n')
		results = []
		
		for i, line in enumerate(lines):
			for pattern in patterns:
				if re.search(pattern, line, re.IGNORECASE):
					start = max(0, i - context_lines)
					end = min(len(lines), i + context_lines + 1)
					context = '\n'.join(lines[start:end])
					results.append((line.strip(), context))
					break
		
		return results

	def fuzzy_match(self, text: str, keywords: List[str], 
				   threshold: float = 0.8) -> List[str]:
		"""
		Perform fuzzy matching using Levenshtein distance.
		
		Args:
			text: Text to search in
			keywords: List of keywords to match
			threshold: Similarity threshold (0.0 to 1.0)
			
		Returns:
			List of matched keywords
		"""
		def levenshtein_distance(s1: str, s2: str) -> int:
			if len(s1) < len(s2):
				return levenshtein_distance(s2, s1)
			if len(s2) == 0:
				return len(s1)
			
			previous_row = list(range(len(s2) + 1))
			for i, c1 in enumerate(s1):
				current_row = [i + 1]
				for j, c2 in enumerate(s2):
					insertions = previous_row[j + 1] + 1
					deletions = current_row[j] + 1
					substitutions = previous_row[j] + (c1 != c2)
					current_row.append(min(insertions, deletions, substitutions))
				previous_row = current_row
			return previous_row[-1]
		
		def similarity(s1: str, s2: str) -> float:
			max_len = max(len(s1), len(s2))
			if max_len == 0:
				return 1.0
			return 1 - levenshtein_distance(s1, s2) / max_len
		
		text_lower = text.lower()
		matches = []
		
		for keyword in keywords:
			keyword_lower = keyword.lower()
			
			# Check exact substring match first
			if keyword_lower in text_lower:
				matches.append(keyword)
				continue
			
			# Check fuzzy match against words in text
			words = re.split(r'\W+', text_lower)
			for word in words:
				if len(word) > 0 and similarity(keyword_lower, word) >= threshold:
					matches.append(keyword)
					break
		
		return matches

	def parse_job_post(self, job_text: str) -> Dict[str, Union[str, List[str], int, Dict, float]]:
		job_text_lower = job_text.lower()
		
		languages_dict = self.extract_languages(job_text_lower)
		language_score = self.calculate_language_score(languages_dict)
		
		result = {
			'title': self.extract_title(job_text),
			'years_experience': self.extract_experience(job_text_lower),
			'programming_languages': languages_dict,
			'language_score': language_score,
			'education': self.extract_education(job_text_lower),
			'sophistication_score': self.extract_sop(job_text),
			'score': self.extract_sop(job_text),  # Alias for simplicity
			'raw_text': job_text
		}
		
		# NEW: Add salary and location extraction
		salary_data = self.extract_salary(job_text_lower)
		location_data = self.extract_location(job_text_lower)
		
		result.update({
			'salary': salary_data,
			'location': location_data,
		})
		
		return result

	def extract_title(self, job_text: str) -> str:
		lines = job_text.strip().split('\n')
		first_line = lines[0].strip() if lines else ""
		
		# Check for explicit title patterns first
		title_patterns = [
			r'job title:?\s*(.+?)(?:\n|$)',
			r'position:?\s*(.+?)(?:\n|$)', 
			r'role:?\s*(.+?)(?:\n|$)',
			r'title:?\s*(.+?)(?:\n|$)'
		]
		
		# Look for structured title in first few lines
		for line in lines[:3]:
			for pattern in title_patterns:
				match = re.search(pattern, line, re.IGNORECASE)
				if match:
					return match.group(1).strip()
		
		# Fall back to existing indicator logic
		title_indicators = ['developer', 'engineer', 'analyst', 'manager', 'lead', 'senior', 'junior', 'intern']
		
		if any(indicator in first_line.lower() for indicator in title_indicators):
			return first_line
		
		for line in lines[:5]:
			if any(indicator in line.lower() for indicator in title_indicators):
				return line.strip()
		
		return first_line

	def extract_experience(self, job_text: str) -> Optional[int]:
		for pattern in self.experience_patterns:
			matches = re.findall(pattern, job_text, re.IGNORECASE)
			for match in matches:
				if isinstance(match, tuple):
					# Handle tuple matches (e.g., from range patterns like "4-6 years")
					nums = [int(group) for group in match if group and group.isdigit()]
					if len(nums) == 2:
						# Take average of range
						return int((nums[0] + nums[1]) / 2)
					elif len(nums) == 1:
						return nums[0]
				elif match and match.isdigit():
					return int(match)
		return None

	def extract_languages(self, job_text: str) -> Dict[str, Dict]:
		"""Extract languages organized by families and frameworks, with required vs preferred distinction."""
		# Split text into sections to identify required vs preferred
		text_lower = job_text.lower()
		
		# Find sections
		required_section = ""
		preferred_section = ""
		
		# Look for section headers
		req_patterns = [
			r'required\s+(?:qualifications?|skills?|experience)(.*?)(?=preferred|nice.to.have|bonus|$)',
			r'must\s+have(.*?)(?=preferred|nice.to.have|bonus|$)',
			r'requirements?(.*?)(?=preferred|nice.to.have|bonus|$)'
		]
		
		pref_patterns = [
			r'(?:preferred|nice.to.have|bonus|plus)\s+(?:qualifications?|skills?|experience)(.*?)$',
			r'(?:preferred|nice.to.have|bonus|plus)(.*?)$'
		]
		
		for pattern in req_patterns:
			match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
			if match:
				required_section = match.group(1)
				break
		
		for pattern in pref_patterns:
			match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
			if match:
				preferred_section = match.group(1)
				break
		
		# If no clear sections found, treat first 70% as required, rest as preferred
		if not required_section and not preferred_section:
			split_point = int(len(text_lower) * 0.7)
			required_section = text_lower[:split_point]
			preferred_section = text_lower[split_point:]
		
		found_families = {}
		
		for family_name, family_data in self.language_families.items():
			req_core, req_frameworks = self._find_tech_in_section(required_section, family_data)
			pref_core, pref_frameworks = self._find_tech_in_section(preferred_section, family_data)
			
			# Only add family if something was found
			if req_core or req_frameworks or pref_core or pref_frameworks:
				found_families[family_name] = {
					'required_core': req_core,
					'required_frameworks': req_frameworks,
					'preferred_core': pref_core,
					'preferred_frameworks': pref_frameworks
				}
		
		return found_families
	
	def _find_tech_in_section(self, section_text: str, family_data: Dict) -> tuple:
		"""Helper to find technologies in a text section."""
		found_core = []
		found_frameworks = []
		
		# Check core languages
		for lang in family_data['core']:
			pattern = r'\b' + re.escape(lang) + r'\b'
			if self.pattern_match(section_text, [pattern], 'exists'):
				found_core.append(lang.title())
		
		# Check frameworks
		for framework in family_data['frameworks']:
			pattern = r'\b' + re.escape(framework) + r'\b'
			if self.pattern_match(section_text, [pattern], 'exists'):
				found_frameworks.append(framework.title())
		
		return found_core, found_frameworks

	def calculate_language_score(self, languages_dict: Dict) -> float:
		"""Calculate weighted language score: Required = full points, Preferred = 0.1 points. Capped at 6.0."""
		total_score = 0.0
		
		for family_name, family_data in languages_dict.items():
			# Required: 1 point per family + 0.2 per framework
			if family_data['required_core']:
				total_score += 1.0
			total_score += len(family_data['required_frameworks']) * 0.2
			
			# Preferred: 0.1 point per family + 0.02 per framework
			if family_data['preferred_core'] and not family_data['required_core']:
				total_score += 0.1
			total_score += len(family_data['preferred_frameworks']) * 0.02
		
		# Cap at 6.0 total points
		return min(total_score, 6.0)

	def extract_education(self, job_text: str) -> List[str]:
		matches = self.pattern_match(job_text, self.education_patterns, 'all')
		if matches:
			return list(set(matches))
		return []

	# NEW: Salary extraction method
	def extract_salary(self, job_text: str) -> Dict[str, Union[int, str, None]]:
		"""Extract salary information."""
		
		def parse_salary_value(value_str: str) -> int:
			"""Convert salary string to integer (handles K notation)."""
			if not value_str:
				return 0
			
			# Remove $ and commas
			clean_val = re.sub(r'[$,]', '', value_str.lower())
			
			# Handle K notation
			if clean_val.endswith('k'):
				return int(float(clean_val[:-1]) * 1000)
			
			return int(float(clean_val))
		
		# Try each salary pattern
		for pattern in self.salary_patterns:
			matches = re.findall(pattern, job_text, re.IGNORECASE)
			for match in matches:
				if isinstance(match, tuple) and len(match) >= 2:
					# Range salary (min-max)
					min_sal = parse_salary_value(match[0])
					max_sal = parse_salary_value(match[1])
					if min_sal > 0 and max_sal > 0:
						return {
							'raw_min': min_sal,
							'raw_max': max_sal,
							'raw_avg': (min_sal + max_sal) // 2,
							'type': 'range'
						}
				elif isinstance(match, str):
					# Single salary value
					sal = parse_salary_value(match)
					if sal > 0:
						return {
							'raw_min': sal,
							'raw_max': sal,
							'raw_avg': sal,
							'type': 'single'
						}
		
		return {
			'raw_min': None,
			'raw_max': None,
			'raw_avg': None,
			'type': 'not_found'
		}

	def extract_sop(self, job_text: str) -> int:
		"""Extract sophistication level based on keyword matches."""
		
		# Define sophistication patterns with weights
		sophistication_patterns = {
			'high': [
				r'\b(?:architect|architecture|architectural|system\s+design|design\s+patterns?)\b',
				r'\b(?:mentor|mentoring|coaching)\b',
				r'\b(?:triage|triaging|incident\s+response)\b'
			],
			'medium': [
				r'\b(?:senior|sr\.?|lead|principal|staff|expert)\b',
				r'\b(?:leading|leadership|tech\s+lead|team\s+lead)\b',
				r'\b(?:ownership|accountable|accountability|responsible\s+for|decisions|strategy|roadmap)\b',
				r'\b(?:optimize|optimization|performance|scalable?|scale|distributed|enterprise|production)\b',
				r'\b(?:technical\s+direction|vision|cross-functional|stakeholder)\b'
			],
			'low': [
				r'\b(?:train|training|onboard|onboarding)\b',
				r'\b(?:code\s+review|technical\s+review|peer\s+review|best\s+practices)\b',
				r'\b(?:on-call|escalation)\b'
			]
		}
		
		score = 0
		
		# Check high sophistication patterns (3 points each)
		for pattern in sophistication_patterns['high']:
			matches = re.findall(pattern, job_text, re.IGNORECASE)
			score += len(matches) * 3
		
		# Check medium sophistication patterns (2 points each)
		for pattern in sophistication_patterns['medium']:
			matches = re.findall(pattern, job_text, re.IGNORECASE)
			score += len(matches) * 2
		
		# Check low sophistication patterns (1 point each)
		for pattern in sophistication_patterns['low']:
			matches = re.findall(pattern, job_text, re.IGNORECASE)
			score += len(matches) * 1
		
		if score > 50:
			score = 50  # Cap at 50
		if score < 10:
			score = 10  # Minimum score
	
		return score

	# Keep original working location extraction method
	def extract_location(self, job_text: str) -> Dict[str, Union[str, List[str], bool]]:
		"""Extract location information."""
		
		locations = []
		work_type = None
		
		# Try each location pattern
		for pattern in self.location_patterns:
			matches = re.findall(pattern, job_text, re.IGNORECASE)
			for match in matches:
				if isinstance(match, str) and match:
					location_clean = match.strip().rstrip('.,')
					
					# Check if it's a work type indicator
					work_indicators = ['remote', 'hybrid', 'on-site', 'work from home', 'wfh']
					if any(indicator in location_clean.lower() for indicator in work_indicators):
						if not work_type:  # Only set if not already found
							work_type = location_clean.lower()
					else:
						# It's a location
						if location_clean not in locations:
							locations.append(location_clean)
		
		# Determine primary location
		primary_location = locations[0] if locations else None
		
		# Default work type if not specified
		if not work_type:
			if any(word in job_text for word in ['remote', 'work from home', 'wfh']):
				work_type = 'remote'
			elif primary_location:
				work_type = 'on-site'
		
		return {
			'primary_location': primary_location,
			'all_locations': locations,
			'work_type': work_type,
			'is_remote': work_type and 'remote' in work_type if work_type else False
		}

	# NEW: Salary normalization to SF wages
	def normalize_salary_to_sf(self, salary_data: Dict, location_data: Dict) -> Dict:
		"""Normalize salary to San Francisco cost of living."""
		
		if not salary_data or salary_data['type'] == 'not_found':
			return salary_data
		
		# Determine location for COL adjustment
		location = None
		if location_data.get('is_remote'):
			location = 'remote'
		elif location_data.get('primary_location'):
			location = location_data['primary_location'].lower()
		
		# Find best matching city for COL multiplier
		multiplier = 1.0  # Default to SF rate if no match
		if location:
			# Try exact match first
			if location in self.col_multipliers:
				multiplier = self.col_multipliers[location]
			else:
				# Try partial matching
				for city, mult in self.col_multipliers.items():
					if city in location or location in city:
						multiplier = mult
						break
		
		# Apply normalization
		result = salary_data.copy()
		
		if salary_data['raw_min']:
			result['sf_normalized_min'] = int(salary_data['raw_min'] / multiplier)
		if salary_data['raw_max']:
			result['sf_normalized_max'] = int(salary_data['raw_max'] / multiplier)
		if salary_data['raw_avg']:
			result['sf_normalized_avg'] = int(salary_data['raw_avg'] / multiplier)
		
		result['col_multiplier'] = multiplier
		result['adjustment_location'] = location
		
		return result

	def classify_job_level(self, job_data: Dict) -> Dict[str, Union[str, bool, Dict, float]]:
		"""
		Classifies job requirements using weighted complexity score (language score * sop percentage).
		"""
		years_exp = job_data.get('years_experience', 0) or 0
		language_score = job_data.get('language_score', 0) or 0
		sop_score = job_data.get('sophistication_score', 10) or job_data.get('score', 10)
		
		# Calculate weighted complexity score
		sop_percentage = sop_score / 50.0  # Convert to percentage (max sop is 50)
		weighted_complexity = language_score * sop_percentage
		
		# Determine tier based on years of experience and weighted complexity
		if years_exp <= 2 and weighted_complexity <= 1.5:
			recommended_tier = 'Junior'
		elif years_exp <= 5 and weighted_complexity <= 3.0:
			recommended_tier = 'Middle'
		else:
			recommended_tier = 'Senior'
		
		classification = {
			'language_score': language_score,
			'sophistication_score': sop_score,
			'sophistication_percentage': sop_percentage,
			'weighted_complexity': weighted_complexity,
			'recommended_tier': recommended_tier,
			'recommended_level': None,
			'overall_assessment': None
		}

		# Calculate level based on sophistication score
		if sop_score <= 15:
			classification['recommended_level'] = 'Entry Level'
		elif sop_score <= 22:
			classification['recommended_level'] = 'Knowledgeable'
		elif sop_score <= 30:
			classification['recommended_level'] = 'Standard'
		elif sop_score <= 38:
			classification['recommended_level'] = 'Skilled'
		elif sop_score <= 44:
			classification['recommended_level'] = 'Advanced'
		else:
			classification['recommended_level'] = 'Boundary'

		classification['overall_assessment'] = f"{recommended_tier} {classification['recommended_level']} (Complexity: {weighted_complexity:.2f})"
		
		return classification

	def _check_level_fit(self, years_exp: int, languages: List[str], 
					education: List[str], level: str) -> Dict[str, str]:
					"""Check if candidate meets requirements for a specific level."""
					level_req = self.experience_levels[level]
					
					# Check experience - treat 0/None as junior level (0 years)
					if years_exp is None or years_exp == 0:
						# No experience mentioned = junior level only
						exp_fit = 'match' if level == 'junior' else 'not_match'
					else:
						exp_fit = 'match' if (level_req['min_years'] <= years_exp <= level_req['max_years']) else 'not_match'
					
					# Check languages - treat empty list as minimal requirement (junior level only)
					if not languages:
						# No languages mentioned = junior level only
						lang_fit = 'match' if level == 'junior' else 'not_match'
					else:
						required_langs = self.language_requirements[level]
						lang_fit = 'match' if len(languages) >= required_langs else 'not_match'
					
					# Check education - treat empty list as no degree requirement (junior level only)
					if not education:
						# No education mentioned = junior level only
						edu_fit = 'match' if level == 'junior' else 'not_match'
					else:
						required_edu = self.education_requirements[level]
						has_required_edu = any(
							any(req_edu in edu.lower() for req_edu in required_edu) 
							for edu in education
						)
						edu_fit = 'match' if has_required_edu else 'not_match'
					
					# Overall assessment
					criteria = [exp_fit, lang_fit, edu_fit]
					under_count = criteria.count('match')
					overall = 'match' if under_count >= 2 else 'not_match'
					
					return {
						'experience': exp_fit,
						'languages': lang_fit,
						'education': edu_fit,
						'overall': overall,
						'score': f"{under_count}/3"
					}


def main():
	import argparse
	
	parser_arg = argparse.ArgumentParser(description='Parse job posting from file')
	parser_arg.add_argument('file', help='Path to job posting file')
	args = parser_arg.parse_args()
	
	try:
		with open(args.file, 'r', encoding='utf-8') as f:
			job_text = f.read()
	except FileNotFoundError:
		print(f"Error: File '{args.file}' not found.")
		return
	except Exception as e:
		print(f"Error reading file: {e}")
		return
	
	parser = JobPostParser()
	result = parser.parse_job_post(job_text)
	
	# Normalize salary to SF
	result['salary'] = parser.normalize_salary_to_sf(result['salary'], result['location'])
	
	classification = parser.classify_job_level(result)

	print()

	# Display results
	print("Parsed Job Post:")
	print(f"Title: {result['title']}")
	print(f"Years of Experience: {result['years_experience']}")
	# Display programming languages by family with required/preferred breakdown
	print(f"Programming Languages (Score: {result['language_score']:.1f}):")
	for family, data in result['programming_languages'].items():
		family_display = family.replace('_', ' ').title()
		has_content = any([data['required_core'], data['required_frameworks'], 
						  data['preferred_core'], data['preferred_frameworks']])
		
		if has_content:
			print(f"  {family_display}:")
			
			# Required technologies
			if data['required_core'] or data['required_frameworks']:
				req_parts = []
				if data['required_core']:
					req_parts.append(f"Core: {', '.join(data['required_core'])}")
				if data['required_frameworks']:
					req_parts.append(f"Frameworks: {', '.join(data['required_frameworks'])}")
				print(f"    Required: {' + '.join(req_parts)}")
			
			# Preferred technologies  
			if data['preferred_core'] or data['preferred_frameworks']:
				pref_parts = []
				if data['preferred_core']:
					pref_parts.append(f"Core: {', '.join(data['preferred_core'])}")
				if data['preferred_frameworks']:
					pref_parts.append(f"Frameworks: {', '.join(data['preferred_frameworks'])}")
				print(f"    Preferred: {' + '.join(pref_parts)}")
	
	print(f"Education: {result['education']}")
	
	# NEW: Display salary information
	sal = result['salary']
	if sal['type'] != 'not_found':
		print(f"\nSalary Information:")
		print(f"  Raw Range: ${sal.get('raw_min', 'N/A'):,} - ${sal.get('raw_max', 'N/A'):,}")
		if sal.get('sf_normalized_avg'):
			print(f"  SF Normalized: ${sal.get('sf_normalized_min', 0):,} - ${sal.get('sf_normalized_max', 0):,}")
			print(f"  COL Multiplier: {sal.get('col_multiplier', 1.0):.2f} (based on {sal.get('adjustment_location', 'unknown')})")
	
	# Display location information (keep simple to avoid issues)
	loc = result['location']
	if loc['primary_location'] or loc['work_type']:
		print(f"\nLocation Information:")
		print(f"  Primary Location: {loc['primary_location'] or 'Not specified'}")
		print(f"  Work Type: {loc['work_type'] or 'Not specified'}")
		print(f"  Is Remote: {loc['is_remote']}")
	
	print("\nLevel Classification:")
	print(f"Recommended Tier: {classification.get('recommended_tier', 'N/A')}")
	print(f"Recommended Level: {classification.get('recommended_level', 'N/A')}")
	print(f"Language Score: {classification.get('language_score', 0):.1f}")
	print(f"Sophistication Score: {classification.get('sophistication_score', 0)}/50 ({classification.get('sophistication_percentage', 0):.1%})")
	print(f"Weighted Complexity: {classification.get('weighted_complexity', 0):.2f}")
	print(f"Overall Assessment: {classification['overall_assessment']}")


if __name__ == "__main__":
	main()
import unittest

from tbyc_dataset.extraction.discussion_entities_prompt import (
    ENTITY_CLASSES,
    EXAMPLE_SPECS,
    PROMPT_DESCRIPTION,
    build_langextract_examples,
)


class DiscussionEntityPromptTests(unittest.TestCase):
    def test_example_extractions_are_verbatim_and_in_order(self) -> None:
        for spec in EXAMPLE_SPECS:
            cursor = 0
            for extraction in spec["extractions"]:
                extraction_text = extraction["extraction_text"]
                position = spec["text"].find(extraction_text, cursor)
                self.assertNotEqual(
                    position,
                    -1,
                    msg=f"{spec['name']} has non-verbatim extraction text: {extraction_text!r}",
                )
                cursor = position + len(extraction_text)

    def test_allowed_class_set_matches_expected_schema(self) -> None:
        expected_classes = {
            "Problem Statement",
            "Proposed Solution",
            "Alternative Solution",
            "Design Decision",
            "Trade-off Argument",
            "Rationale",
            "Constraint",
            "Assumption",
            "Implementation Detail",
            "Code Snippet",
            "Algorithm / Approach",
            "API Design",
            "Data Structure Choice",
            "Configuration Choice",
            "Benchmark Result",
            "Performance Claim",
            "Test Case",
            "Bug Reproduction Steps",
            "Edge Case",
            "Empirical Evidence",
            "Question",
            "Answer / Clarification",
            "Agreement",
            "Disagreement",
            "Suggestion",
            "Critique",
            "Task Assignment",
            "Status Update",
            "Priority Discussion",
            "Blocking Issue",
            "Dependency",
            "Reference (Code)",
            "Reference (Pull Request)",
            "Reference (External)",
        }
        self.assertEqual(set(ENTITY_CLASSES), expected_classes)

    def test_examples_cover_major_discussion_modes_and_prompt_rules(self) -> None:
        classes = {
            extraction["extraction_class"]
            for spec in EXAMPLE_SPECS
            for extraction in spec["extractions"]
        }
        self.assertIn("Problem Statement", classes)
        self.assertIn("Proposed Solution", classes)
        self.assertIn("Design Decision", classes)
        self.assertIn("Trade-off Argument", classes)
        self.assertIn("Constraint", classes)
        self.assertIn("Implementation Detail", classes)
        self.assertIn("Benchmark Result", classes)
        self.assertIn("Question", classes)
        self.assertIn("Answer / Clarification", classes)
        self.assertIn("Agreement", classes)
        self.assertIn("Task Assignment", classes)
        self.assertIn("Reference (Pull Request)", classes)
        self.assertIn("Ignore greetings, thanks, social chatter", PROMPT_DESCRIPTION)
        self.assertIn("author: comment", PROMPT_DESCRIPTION)

    def test_examples_use_structured_specific_attributes(self) -> None:
        for spec in EXAMPLE_SPECS:
            for extraction in spec["extractions"]:
                attributes = extraction["attributes"]
                self.assertIn("speaker", attributes)
                self.assertIn("topic", attributes)
                self.assertIn("factors", attributes)
                self.assertIsInstance(attributes["factors"], list)
                self.assertTrue(attributes["factors"])
                self.assertGreaterEqual(len(attributes.keys()), 4)

    def test_runtime_examples_build_for_all_specs(self) -> None:
        class FakeExtraction:
            def __init__(self, extraction_class, extraction_text, attributes):
                self.extraction_class = extraction_class
                self.extraction_text = extraction_text
                self.attributes = attributes

        class FakeExampleData:
            def __init__(self, text, extractions):
                self.text = text
                self.extractions = extractions

        class FakeLX:
            class data:
                Extraction = FakeExtraction
                ExampleData = FakeExampleData

        runtime_examples = build_langextract_examples(FakeLX)
        self.assertTrue(runtime_examples)
        self.assertTrue(all(example.extractions for example in runtime_examples))
        self.assertEqual(len(runtime_examples), len(EXAMPLE_SPECS))

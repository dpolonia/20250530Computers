"""
Functional tests for the paper revision workflow.

These tests verify that the paper revision tool works correctly for complete user workflows.
"""

import os
import pytest
import tempfile
from unittest.mock import patch, Mock

from src.core.paper_revision_tool import PaperRevisionTool


class TestPaperRevisionWorkflow:
    """Tests for complete paper revision workflows."""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as paper_file:
            paper_path = paper_file.name
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as reviewer1_file:
            reviewer1_file.write(b"Reviewer 1 comments")
            reviewer1_path = reviewer1_file.name
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as reviewer2_file:
            reviewer2_file.write(b"Reviewer 2 comments")
            reviewer2_path = reviewer2_file.name
        
        # Create temporary output directory
        output_dir = tempfile.mkdtemp()
        
        # Yield the paths
        yield {
            "paper_path": paper_path,
            "reviewer_paths": [reviewer1_path, reviewer2_path],
            "output_dir": output_dir
        }
        
        # Clean up
        os.unlink(paper_path)
        os.unlink(reviewer1_path)
        os.unlink(reviewer2_path)
        # We don't remove the output_dir to allow inspection after tests
    
    @pytest.fixture
    def mock_services(self):
        """Mock the services used by the PaperRevisionTool."""
        # Create mock service factory with mock services
        mock_factory = Mock()
        
        # Mock paper service
        mock_paper_service = Mock()
        mock_paper_service.analyze_paper.return_value = Mock(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="Test abstract",
            sections={
                "Introduction": "Test introduction",
                "Methods": "Test methods",
                "Results": "Test results",
                "Discussion": "Test discussion"
            },
            references=[Mock(key="ref1", title="Reference 1")]
        )
        
        # Mock reviewer service
        mock_reviewer_service = Mock()
        mock_reviewer_service.analyze_reviewer_comments.return_value = [
            Mock(
                reviewer_number=1,
                overall_assessment="minor_revision",
                main_concerns=["Methods lack clarity", "Missing references"],
                text="Reviewer 1 comments"
            ),
            Mock(
                reviewer_number=2,
                overall_assessment="major_revision",
                main_concerns=["Insufficient analysis", "Weak conclusion"],
                text="Reviewer 2 comments"
            )
        ]
        mock_reviewer_service.analyze_editor_requirements.return_value = {
            "decision": "revise",
            "key_requirements": ["Address reviewer concerns", "Improve methods section"]
        }
        
        # Mock solution service
        mock_solution_service = Mock()
        mock_solution_service.identify_issues.return_value = [
            Mock(
                description="Methods section lacks clarity",
                type="methodology",
                severity="high",
                reviewer_number=1,
                section="Methods"
            ),
            Mock(
                description="Missing recent references",
                type="references",
                severity="medium",
                reviewer_number=1,
                section="References"
            )
        ]
        mock_solution_service.generate_solutions.return_value = [
            Mock(
                description="Improve methods section clarity",
                approach="Add more details about the methodology",
                impact="High"
            ),
            Mock(
                description="Add missing recent references",
                approach="Include references from the last 2 years",
                impact="Medium"
            )
        ]
        mock_solution_service.generate_specific_changes.return_value = [
            Mock(
                section="Methods",
                old_text="Test methods",
                new_text="Test methods with improved clarity and additional details",
                reason="To address the reviewer's concern about clarity"
            ),
            Mock(
                section="References",
                old_text="References section",
                new_text="References section with additional recent works",
                reason="To address the reviewer's concern about missing references"
            )
        ]
        
        # Mock reference service
        mock_reference_service = Mock()
        mock_reference_service.validate_and_update_references.return_value = [
            Mock(
                key="new_ref1",
                title="New Reference 1",
                authors=["New Author 1"],
                year=2023
            ),
            Mock(
                key="new_ref2",
                title="New Reference 2",
                authors=["New Author 2"],
                year=2024
            )
        ]
        
        # Mock document service
        mock_document_service = Mock()
        mock_document_service.create_changes_document.return_value = "/path/to/changes.docx"
        mock_document_service.create_revised_paper.return_value = "/path/to/revised_paper.docx"
        mock_document_service.create_assessment.return_value = "/path/to/assessment.docx"
        mock_document_service.create_editor_letter.return_value = "/path/to/editor_letter.docx"
        
        # Set up factory to return mock services
        mock_factory.get_paper_service.return_value = mock_paper_service
        mock_factory.get_reviewer_service.return_value = mock_reviewer_service
        mock_factory.get_solution_service.return_value = mock_solution_service
        mock_factory.get_reference_service.return_value = mock_reference_service
        mock_factory.get_document_service.return_value = mock_document_service
        
        return mock_factory
    
    @patch('src.core.paper_revision_tool.ServiceFactory')
    def test_full_revision_process(self, mock_factory_class, temp_files, mock_services):
        """Test the full paper revision process."""
        # Arrange
        mock_factory_class.return_value = mock_services
        
        tool = PaperRevisionTool(
            original_paper_path=temp_files["paper_path"],
            reviewer_comment_files=temp_files["reviewer_paths"],
            output_dir=temp_files["output_dir"],
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            optimize_costs=True
        )
        
        # Act
        results = tool.run_full_process()
        
        # Assert
        assert len(results) == 4
        assert "changes" in results
        assert "revised_paper" in results
        assert "assessment" in results
        assert "editor_letter" in results
        
        # Verify that all services were called
        paper_service = mock_services.get_paper_service.return_value
        paper_service.analyze_paper.assert_called_once()
        
        reviewer_service = mock_services.get_reviewer_service.return_value
        reviewer_service.analyze_reviewer_comments.assert_called_once()
        reviewer_service.analyze_editor_requirements.assert_called_once()
        
        solution_service = mock_services.get_solution_service.return_value
        solution_service.identify_issues.assert_called_once()
        solution_service.generate_solutions.assert_called_once()
        solution_service.generate_specific_changes.assert_called_once()
        
        reference_service = mock_services.get_reference_service.return_value
        reference_service.validate_and_update_references.assert_called_once()
        
        document_service = mock_services.get_document_service.return_value
        document_service.create_changes_document.assert_called_once()
        document_service.create_revised_paper.assert_called_once()
        document_service.create_assessment.assert_called_once()
        document_service.create_editor_letter.assert_called_once()
    
    @patch('src.core.paper_revision_tool.ServiceFactory')
    def test_interactive_mode_commands(self, mock_factory_class, temp_files, mock_services):
        """Test individual commands in interactive mode."""
        # Arrange
        mock_factory_class.return_value = mock_services
        
        tool = PaperRevisionTool(
            original_paper_path=temp_files["paper_path"],
            reviewer_comment_files=temp_files["reviewer_paths"],
            output_dir=temp_files["output_dir"],
            provider="anthropic",
            model_name="claude-3-opus-20240229"
        )
        
        # Act - Test analyze_paper command
        paper = tool.analyze_paper()
        
        # Assert
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert len(paper.sections) == 4
        
        # Act - Test analyze_reviewer_comments command
        reviewer_comments = tool.analyze_reviewer_comments()
        
        # Assert
        assert len(reviewer_comments) == 2
        assert reviewer_comments[0].reviewer_number == 1
        assert reviewer_comments[1].reviewer_number == 2
        
        # Act - Test identify_issues command
        issues = tool.identify_issues()
        
        # Assert
        assert len(issues) == 2
        assert issues[0].description == "Methods section lacks clarity"
        assert issues[1].description == "Missing recent references"
        
        # Act - Test generate_solutions command
        solutions = tool.generate_solutions()
        
        # Assert
        assert len(solutions) == 2
        assert solutions[0].description == "Improve methods section clarity"
        assert solutions[1].description == "Add missing recent references"
        
        # Act - Test generate_specific_changes command
        changes = tool.generate_specific_changes()
        
        # Assert
        assert len(changes) == 2
        assert changes[0].section == "Methods"
        assert changes[1].section == "References"
        
        # Act - Test validate_and_update_references command
        new_references = tool.validate_and_update_references()
        
        # Assert
        assert len(new_references) == 2
        assert new_references[0].key == "new_ref1"
        assert new_references[1].key == "new_ref2"
        
        # Act - Test create_changes_document command
        changes_doc = tool.create_changes_document()
        
        # Assert
        assert changes_doc == "/path/to/changes.docx"
        
        # Act - Test create_revised_paper command
        revised_paper = tool.create_revised_paper()
        
        # Assert
        assert revised_paper == "/path/to/revised_paper.docx"
        
        # Act - Test create_assessment command
        assessment = tool.create_assessment()
        
        # Assert
        assert assessment == "/path/to/assessment.docx"
        
        # Act - Test create_editor_letter command
        editor_letter = tool.create_editor_letter()
        
        # Assert
        assert editor_letter == "/path/to/editor_letter.docx"
    
    @patch('src.core.paper_revision_tool.ServiceFactory')
    def test_error_handling(self, mock_factory_class, temp_files, mock_services):
        """Test error handling in the revision process."""
        # Arrange
        mock_factory_class.return_value = mock_services
        
        # Set up a service to raise an exception
        solution_service = mock_services.get_solution_service.return_value
        solution_service.identify_issues.side_effect = Exception("Test error")
        
        tool = PaperRevisionTool(
            original_paper_path=temp_files["paper_path"],
            reviewer_comment_files=temp_files["reviewer_paths"],
            output_dir=temp_files["output_dir"],
            provider="anthropic",
            model_name="claude-3-opus-20240229"
        )
        
        # Act/Assert - Running the full process should raise an exception
        with pytest.raises(Exception) as excinfo:
            tool.run_full_process()
        
        assert "Test error" in str(excinfo.value)
        
        # Verify that services up to the error were called
        paper_service = mock_services.get_paper_service.return_value
        paper_service.analyze_paper.assert_called_once()
        
        reviewer_service = mock_services.get_reviewer_service.return_value
        reviewer_service.analyze_reviewer_comments.assert_called_once()
        
        # Verify that services after the error were not called
        reference_service = mock_services.get_reference_service.return_value
        reference_service.validate_and_update_references.assert_not_called()
        
        document_service = mock_services.get_document_service.return_value
        document_service.create_changes_document.assert_not_called()
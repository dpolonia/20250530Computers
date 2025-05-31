"""
Document service implementation.

This module implements the document service interface, providing functionality for
creating various documents related to the paper revision process.
"""

import logging
import os
import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.utils.path_utils import (
    construct_output_path, get_filename
)
from src.utils.text_utils import (
    normalize_newlines
)

from src.core.context import RevisionContext
from src.domain.paper import Paper
from src.domain.reviewer_comment import ReviewerComment
from src.domain.change import Change
from src.domain.assessment import Assessment, ImpactLevel
from src.services.interfaces import DocumentServiceInterface
from src.adapters.docx_adapter import DocxAdapter


class DocumentService(DocumentServiceInterface):
    """
    Service for document creation.
    
    This service is responsible for creating various documents related to the
    paper revision process, including changes documents, revised papers, assessment
    documents, and editor letters.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the document service.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
        self.docx_adapter = DocxAdapter(context)
    
    def create_changes_document(
        self,
        changes: List[Change],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a document detailing all changes.
        
        Args:
            changes: List of Change domain entities
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating changes document")
        
        if not changes:
            self.logger.warning("No changes to document")
            return ""
        
        # Determine output path if not provided
        if not output_path:
            output_path = construct_output_path(
                "changes", 
                self.context.output_dir, 
                self.context.original_paper_path, 
                ".docx"
            )
        
        # Organize changes by section and solution
        changes_by_section = {}
        for change in changes:
            section = change.section or "General"
            if section not in changes_by_section:
                changes_by_section[section] = []
            changes_by_section[section].append(change)
        
        # Create changes document using the adapter
        changes_tuples = [(change.old_text, change.new_text, change.reason, None) for change in changes]
        document_path = self.docx_adapter.create_changes_document(changes_tuples, output_path)
        
        self.logger.info(f"Changes document created at {document_path}")
        return document_path
    
    def create_revised_paper(
        self,
        changes: List[Change],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create revised paper with track changes.
        
        Args:
            changes: List of Change domain entities
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating revised paper with track changes")
        
        if not changes:
            self.logger.warning("No changes to apply")
            return ""
        
        # Determine output path if not provided
        if not output_path:
            # Get original paper filename without extension
            paper_name = get_filename(self.context.original_paper_path, with_extension=False)
            output_path = construct_output_path(
                f"{paper_name}_revised", 
                self.context.output_dir, 
                self.context.original_paper_path, 
                ".docx"
            )
        
        # Get the docx version of the original paper
        original_docx = self.context.original_paper_docx
        if not original_docx:
            # Convert PDF to DOCX if needed
            from src.adapters.pdf_adapter import PDFAdapter
            pdf_adapter = PDFAdapter(self.context)
            original_docx = pdf_adapter.pdf_to_docx(self.context.original_paper_path)
            self.context.original_paper_docx = original_docx
        
        # Create a new DOCX adapter with the original paper
        docx_adapter = DocxAdapter(self.context, original_docx)
        
        # Apply each change with track changes
        for change in changes:
            docx_adapter.add_tracked_change(
                old_text=change.old_text,
                new_text=change.new_text,
                reason=change.reason
            )
        
        # Save the revised paper
        document_path = docx_adapter.save(output_path)
        
        self.logger.info(f"Revised paper created at {document_path}")
        return document_path
    
    def create_assessment(
        self,
        changes: List[Change],
        paper: Paper,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create assessment document.
        
        Args:
            changes: List of Change domain entities
            paper: Paper domain entity
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating assessment document")
        
        # Determine output path if not provided
        if not output_path:
            output_path = construct_output_path(
                "assessment", 
                self.context.output_dir, 
                self.context.original_paper_path, 
                ".docx"
            )
        
        # Generate the assessment
        assessment = self._generate_assessment(changes, paper)
        
        # Create assessment document
        docx_adapter = DocxAdapter(self.context)
        
        # Create content for the assessment document
        content = normalize_newlines(f"""
        # Assessment of Paper Revisions
        
        ## Paper Title
        {paper.title}
        
        ## Authors
        {", ".join(paper.authors)}
        
        ## Overall Impact of Changes
        {assessment.overall_impact}
        
        ## Reviewer Concerns Addressed
        {assessment.reviewer_concerns}
        
        ## Areas Strengthened
        {chr(10).join(["- " + area for area in assessment.strengthened_areas])}
        
        ## Remaining Issues
        {chr(10).join(["- " + issue for issue in assessment.remaining_issues])}
        
        ## Recommendations for Future Work
        {chr(10).join(["- " + rec for rec in assessment.recommendations])}
        
        ## Assessment Date
        {assessment.created_at}
        """)
        
        # Write the assessment document
        document_path = docx_adapter.write(content, output_path)
        
        self.logger.info(f"Assessment document created at {document_path}")
        return document_path
    
    def create_editor_letter(
        self,
        reviewer_comments: List[ReviewerComment],
        changes: List[Change],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create letter to the editor.
        
        Args:
            reviewer_comments: List of ReviewerComment domain entities
            changes: List of Change domain entities
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating letter to the editor")
        
        # Determine output path if not provided
        if not output_path:
            output_path = construct_output_path(
                "editor_letter", 
                self.context.output_dir, 
                self.context.original_paper_path, 
                ".docx"
            )
        
        # Group changes by reviewer
        changes_by_reviewer = {}
        for change in changes:
            if hasattr(change.solution, 'issues') and change.solution.issues:
                for issue in change.solution.issues:
                    if hasattr(issue, 'reviewer_number') and issue.reviewer_number:
                        reviewer_number = issue.reviewer_number
                        if reviewer_number not in changes_by_reviewer:
                            changes_by_reviewer[reviewer_number] = []
                        changes_by_reviewer[reviewer_number].append(change)
        
        # Prepare reviewer responses data
        reviewer_responses = []
        for reviewer in reviewer_comments:
            reviewer_number = reviewer.reviewer_number
            reviewer_changes = changes_by_reviewer.get(reviewer_number, [])
            
            # Group reviewer comments by type
            comments_by_type = {
                "methodology": reviewer.methodology_comments,
                "results": reviewer.results_comments,
                "writing": reviewer.writing_comments,
                "references": reviewer.references_comments
            }
            
            # Create response for this reviewer
            response = {
                "reviewer_number": reviewer_number,
                "overall_assessment": reviewer.overall_assessment.value,
                "main_concerns": reviewer.main_concerns,
                "comments_by_type": comments_by_type,
                "changes_made": [(change.reason, change.section) for change in reviewer_changes],
                "text_preview": reviewer.text_preview
            }
            
            reviewer_responses.append(response)
        
        # Create process summary
        process_summary = {
            "paper_title": self.context.paper_title,
            "authors": self.context.paper_authors,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "total_changes": len(changes),
            "reviewer_count": len(reviewer_comments)
        }
        
        # Create editor letter using the adapter
        document_path = self.docx_adapter.create_editor_letter(reviewer_responses, output_path, process_summary)
        
        self.logger.info(f"Editor letter created at {document_path}")
        return document_path
    
    def _generate_assessment(self, changes: List[Change], paper: Paper) -> Assessment:
        """
        Generate an assessment of the changes made to the paper.
        
        Args:
            changes: List of Change domain entities
            paper: Paper domain entity
            
        Returns:
            Assessment domain entity
        """
        # Group changes by type and section
        changes_by_section = {}
        change_types = set()
        
        for change in changes:
            section = change.section or "General"
            if section not in changes_by_section:
                changes_by_section[section] = []
            changes_by_section[section].append(change)
            
            # Track change types based on solutions
            if change.solution and hasattr(change.solution, 'issues') and change.solution.issues:
                for issue in change.solution.issues:
                    if hasattr(issue, 'type'):
                        change_types.add(issue.type.value)
        
        # Count changes by type
        change_counts = {
            "methodology": sum(1 for c in changes if c.solution and any(i.type.value == "methodology" for i in c.solution.issues)),
            "results": sum(1 for c in changes if c.solution and any(i.type.value == "results" for i in c.solution.issues)),
            "writing": sum(1 for c in changes if c.solution and any(i.type.value == "writing" for i in c.solution.issues)),
            "references": sum(1 for c in changes if c.solution and any(i.type.value == "references" for i in c.solution.issues)),
            "structure": sum(1 for c in changes if c.solution and any(i.type.value == "structure" for i in c.solution.issues))
        }
        
        # Determine impact level based on changes
        if len(changes) > 20 or len(change_types) >= 4:
            impact_level = ImpactLevel.HIGH
        elif len(changes) > 10 or len(change_types) >= 3:
            impact_level = ImpactLevel.MEDIUM
        else:
            impact_level = ImpactLevel.LOW
        
        # Identify strengthened areas
        strengthened_areas = []
        if change_counts["methodology"] > 3:
            strengthened_areas.append("Methodology has been significantly improved with more detailed descriptions and justifications")
        elif change_counts["methodology"] > 0:
            strengthened_areas.append("Methodology has been clarified with additional details")
            
        if change_counts["results"] > 3:
            strengthened_areas.append("Results presentation has been substantially enhanced with improved clarity and additional analyses")
        elif change_counts["results"] > 0:
            strengthened_areas.append("Results have been clarified and better explained")
            
        if change_counts["writing"] > 5:
            strengthened_areas.append("Overall writing quality has been significantly improved throughout the paper")
        elif change_counts["writing"] > 0:
            strengthened_areas.append("Writing clarity has been improved in several sections")
            
        if change_counts["references"] > 3:
            strengthened_areas.append("Reference section has been substantially updated with current and relevant citations")
        elif change_counts["references"] > 0:
            strengthened_areas.append("References have been updated with additional relevant citations")
            
        if change_counts["structure"] > 2:
            strengthened_areas.append("Paper structure has been reorganized for improved flow and clarity")
        
        # Identify any remaining issues
        remaining_issues = []
        if not any(section.name.lower() == "limitations" for section in paper.sections):
            remaining_issues.append("The paper would benefit from a dedicated limitations section")
            
        if len(paper.references) < 20:
            remaining_issues.append("The reference list is still relatively limited and could be expanded further")
            
        if not strengthened_areas:
            remaining_issues.append("The changes made may not fully address all reviewer concerns")
            
        # Generate recommendations
        recommendations = []
        if "limitations" in remaining_issues[0] if remaining_issues else "":
            recommendations.append("Add a dedicated limitations section discussing the constraints of the study")
            
        if any("reference" in issue for issue in remaining_issues):
            recommendations.append("Further expand the literature review with additional recent references")
            
        recommendations.append("Consider adding a visual abstract or graphical summary of the key findings")
        recommendations.append("Ensure consistent terminology throughout all sections of the paper")
        
        # Generate overall impact statement
        if impact_level == ImpactLevel.HIGH:
            overall_impact = (
                f"The revisions have substantially improved the paper across {len(change_types)} key areas, "
                f"with a total of {len(changes)} specific changes. The most significant improvements "
                f"are in {', '.join(sorted(change_types, key=lambda x: -change_counts[x])[:2])}. "
                f"These changes collectively address the major concerns raised during review."
            )
        elif impact_level == ImpactLevel.MEDIUM:
            overall_impact = (
                f"The revisions have meaningfully improved the paper, addressing concerns across "
                f"{len(change_types)} areas with {len(changes)} specific changes. The paper is stronger "
                f"particularly in {', '.join(sorted(change_types, key=lambda x: -change_counts[x])[:1])}, "
                f"though some minor issues remain that could be addressed in future revisions."
            )
        else:
            overall_impact = (
                f"The revisions have made targeted improvements to the paper, with {len(changes)} specific changes "
                f"addressing reviewer feedback. While these changes enhance the paper, additional revisions may be "
                f"needed to fully address all reviewer concerns."
            )
        
        # Generate reviewer concerns addressed statement
        reviewer_concerns = (
            f"The revisions address reviewer concerns regarding "
            f"{', '.join(sorted(change_types, key=lambda x: -change_counts[x]))}. "
            f"Specific changes include improvements to "
            f"{', '.join(f'{section} ({len(changes)})' for section, changes in list(changes_by_section.items())[:3])}."
        )
        
        # Create the Assessment domain entity
        return Assessment(
            overall_impact=overall_impact,
            reviewer_concerns=reviewer_concerns,
            remaining_issues=remaining_issues,
            strengthened_areas=strengthened_areas,
            recommendations=recommendations,
            impact_level=impact_level,
            created_at=datetime.datetime.now().strftime("%Y-%m-%d")
        )
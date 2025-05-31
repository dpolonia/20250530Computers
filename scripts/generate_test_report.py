#!/usr/bin/env python3
"""
Test report generator for the Paper Revision System.

This script runs pytest with coverage and generates HTML and JSON reports.
"""

import os
import sys
import json
import subprocess
import datetime
from pathlib import Path


def run_tests_with_coverage():
    """Run tests with coverage and return the result."""
    print("Running tests with coverage...")
    result = subprocess.run(
        ["pytest", "--cov=src", "--cov-report=xml", "--cov-report=html"],
        capture_output=True,
        text=True
    )
    return result


def parse_coverage_data():
    """Parse coverage data from coverage.xml."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse('coverage.xml')
        root = tree.getroot()
        
        coverage_data = {
            'total': float(root.attrib.get('line-rate', 0)) * 100,
            'modules': {}
        }
        
        for package in root.findall('.//package'):
            package_name = package.attrib.get('name', 'unknown')
            coverage_data['modules'][package_name] = {
                'coverage': float(package.attrib.get('line-rate', 0)) * 100,
                'classes': {}
            }
            
            for class_elem in package.findall('.//class'):
                class_name = class_elem.attrib.get('name', 'unknown')
                coverage_data['modules'][package_name]['classes'][class_name] = {
                    'coverage': float(class_elem.attrib.get('line-rate', 0)) * 100
                }
        
        return coverage_data
    except Exception as e:
        print(f"Error parsing coverage data: {e}")
        return None


def generate_report(test_result, coverage_data):
    """Generate a test report with coverage information."""
    now = datetime.datetime.now()
    
    report = {
        'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
        'test_summary': {
            'passed': 'error' not in test_result.stdout.lower(),
            'output': test_result.stdout
        },
        'coverage': coverage_data
    }
    
    # Write JSON report
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Write HTML report
    with open('test_report.html', 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {now.strftime('%Y-%m-%d')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ padding: 10px; background-color: #f0f0f0; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .coverage-section {{ margin-top: 20px; }}
        .module {{ margin-bottom: 10px; padding: 10px; background-color: #f9f9f9; }}
        .progress {{ width: 200px; height: 20px; background-color: #e0e0e0; border-radius: 5px; overflow: hidden; }}
        .progress-bar {{ height: 100%; background-color: #4CAF50; }}
        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; }}
    </style>
</head>
<body>
    <h1>Test Report</h1>
    <div class="summary">
        <p><strong>Date:</strong> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Status:</strong> <span class="{'passed' if report['test_summary']['passed'] else 'failed'}">
            {'PASSED' if report['test_summary']['passed'] else 'FAILED'}
        </span></p>
        <p><strong>Overall Coverage:</strong> {report['coverage']['total']:.2f}%</p>
        <div class="progress">
            <div class="progress-bar" style="width: {report['coverage']['total']}%;"></div>
        </div>
    </div>
    
    <div class="coverage-section">
        <h2>Coverage by Module</h2>
""")
        
        # Add module coverage
        for module_name, module_data in report['coverage']['modules'].items():
            f.write(f"""
        <div class="module">
            <h3>{module_name}</h3>
            <p>Coverage: {module_data['coverage']:.2f}%</p>
            <div class="progress">
                <div class="progress-bar" style="width: {module_data['coverage']}%;"></div>
            </div>
        </div>
""")
        
        # Add test output
        f.write(f"""
    </div>
    
    <h2>Test Output</h2>
    <pre>{report['test_summary']['output']}</pre>
    
    <p>Full coverage details available in the <a href="coverage_html_report/index.html">Coverage Report</a>.</p>
</body>
</html>
""")
    
    print(f"Test report generated: test_report.json and test_report.html")
    print(f"Coverage HTML report available at: coverage_html_report/index.html")


def main():
    """Main function to run tests and generate report."""
    # Set working directory to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run tests with coverage
    test_result = run_tests_with_coverage()
    
    print(test_result.stdout)
    if test_result.stderr:
        print("ERRORS:")
        print(test_result.stderr)
    
    # Parse coverage data
    coverage_data = parse_coverage_data()
    
    if coverage_data:
        # Generate test report
        generate_report(test_result, coverage_data)
    else:
        print("Failed to generate coverage report.")
        return 1
    
    # Return success if tests passed
    return 0 if 'error' not in test_result.stdout.lower() else 1


if __name__ == "__main__":
    sys.exit(main())
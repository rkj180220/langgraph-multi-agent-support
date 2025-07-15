"""
Main application entry point for the multi-agent support system.
Enhanced with Rich and Typer for beautiful CLI experience.
"""

import asyncio
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box
import time

from src.hierarchical_multi_agent_support.system import MultiAgentSupportSystem

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="multi-agent-support",
    help="🤖 Hierarchical Multi-Agent Support System with AWS Bedrock/Claude",
    add_completion=False,
    rich_markup_mode="rich"
)

# Color scheme
COLORS = {
    "primary": "cyan",
    "secondary": "green",
    "accent": "yellow",
    "error": "red",
    "success": "green",
    "info": "cyan",  # Changed from blue to cyan for better visibility
    "warning": "yellow"
}


def create_header() -> Panel:
    """Create a beautiful header for the application."""
    header_text = Text.assemble(
        ("🤖 ", "bold blue"),
        ("Multi-Agent Support System", "bold white"),
        ("\nPowered by AWS Bedrock & Claude", "dim white")
    )
    return Panel(
        Align.center(header_text),
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2)
    )


def create_welcome_message() -> Panel:
    """Create a welcome message panel."""
    welcome_md = """
## Welcome! 👋

I can help you with **IT** and **Finance** queries using advanced AI agents.

### Available Commands:
- `help` - Show available commands
- `info` - Display system information  
- `quit`, `exit`, `bye` - Exit the application

### Example Queries:
- *How do I reset my password?*
- *My computer won't start*
- *How do I submit an expense report?*
- *What's the budget approval process?*
"""
    return Panel(
        Markdown(welcome_md),
        title="[bold cyan]Getting Started[/bold cyan]",
        border_style="green"
    )


def show_system_info(system: MultiAgentSupportSystem) -> None:
    """Display system information in a beautiful format."""
    info = system.get_system_info()

    # Create main info table
    table = Table(title="🔧 System Configuration", box=box.ROUNDED)
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("Model", f"[green]{info['config']['model']}[/green]")
    table.add_row("Region", f"[yellow]{info['config']['region']}[/yellow]")
    table.add_row("Temperature", f"[blue]{info['config']['temperature']}[/blue]")
    table.add_row("Max Tokens", f"[magenta]{info['config']['max_tokens']}[/magenta]")

    # Create agents table
    agents_table = Table(title="🤖 Available Agents", box=box.ROUNDED)
    agents_table.add_column("Agent Type", style="cyan")
    agents_table.add_column("Name", style="green")

    for agent_type, agent_name in info['agents'].items():
        agents_table.add_row(agent_type.replace("_", " ").title(), agent_name)

    # Create tools table
    tools_table = Table(title="🛠️ Available Tools", box=box.ROUNDED)
    tools_table.add_column("Tool", style="yellow")
    tools_table.add_column("Status", style="green")

    for tool in info['tools']:
        tools_table.add_row(tool.replace("_", " ").title(), "✅ Available")

    # Display all tables
    console.print(table)
    console.print()
    console.print(agents_table)
    console.print()
    console.print(tools_table)


def show_help() -> None:
    """Display help information."""
    help_md = """
## 📋 Available Commands

### Basic Commands:
- `help` - Show this help message
- `info` - Display system information
- `quit`, `exit`, `bye` - Exit the application

### Query Examples:

#### 🔧 IT Support:
- *How do I reset my password?*
- *My computer won't start*
- *I can't connect to the network*
- *How do I install new software?*
- *My email is not working*

#### 💰 Finance Support:
- *How do I submit an expense report?*
- *What's the budget approval process?*
- *How do I process vendor payments?*
- *What are the expense policy limits?*
- *How do I generate financial reports?*

### Tips:
- Be specific with your queries
- Mention whether it's an IT or Finance issue
- Provide context about your problem
"""
    console.print(Panel(Markdown(help_md), title="[bold cyan]Help[/bold cyan]", border_style="blue"))


def format_agent_response(result: dict) -> None:
    """Format and display agent response beautifully."""
    if result["success"]:
        # Success response
        response_panel = Panel(
            result["response"],
            title="[bold green]✅ Response[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        console.print(response_panel)

        # Metadata if available
        if result.get("metadata"):
            metadata_table = Table(title="📊 Processing Details", box=box.SIMPLE)
            metadata_table.add_column("Detail", style="cyan")
            metadata_table.add_column("Value", style="white")

            for key, value in result["metadata"].items():
                display_key = key.replace("_", " ").title()

                if key == "processing_path":
                    # Format processing path as a visual flow
                    path_str = " → ".join(value)
                    value = f"[green]{path_str}[/green]"
                elif key == "routing_decision":
                    value = f"[yellow]{value}[/yellow]"
                elif key == "tools_used":
                    value = f"[blue]{value}[/blue]"
                elif key == "specialist_agents":
                    if value:
                        value = f"[cyan]{', '.join(value)}[/cyan]"
                    else:
                        continue  # Skip empty specialist agents
                elif key == "evaluation_success":
                    value = f"[green]✅ Yes[/green]" if value else f"[red]❌ No[/red]"
                elif key == "evaluated":
                    value = f"[green]✅ Yes[/green]" if value else f"[red]❌ No[/red]"
                elif key == "total_processing_steps":
                    value = f"[magenta]{value} steps[/magenta]"

                metadata_table.add_row(display_key, str(value))

            console.print()
            console.print(metadata_table)
    else:
        # Error response - check if it's a helpful unclear query response
        response_text = result.get("response", "")
        if "I can only help with IT or Finance related queries" in response_text:
            console.print(Panel(
                response_text,
                title="[bold yellow]🤔 Need More Information[/bold yellow]",
                border_style="yellow",
                padding=(1, 2)
            ))
        else:
            # Technical error
            console.print(Panel(
                f"[red]Error: {result.get('error', 'Unknown error')}[/red]\n\n{result.get('response', '')}",
                title="[bold red]❌ Error[/bold red]",
                border_style="red",
                padding=(1, 2)
            ))

        # Show processing path even for errors if available
        if result.get("metadata") and result["metadata"].get("processing_path"):
            path_str = " → ".join(result["metadata"]["processing_path"])
            console.print(f"\n[dim]Processing Path: {path_str}[/dim]")


async def interactive_mode(system: MultiAgentSupportSystem) -> None:
    """Run the enhanced interactive mode."""
    console.clear()

    # Show header
    console.print(create_header())
    console.print()

    # Show welcome message
    console.print(create_welcome_message())
    console.print()

    while True:
        try:
            # Get user input with Rich prompt
            user_input = Prompt.ask(
                "\n[bold cyan]💬 Your query[/bold cyan]",
                default="",
                show_default=False
            ).strip()

            # Handle empty input
            if not user_input:
                console.print("[yellow]⚠️ Please enter a query or command.[/yellow]")
                continue

            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print(Panel(
                    "[bold green]👋 Thank you for using the Multi-Agent Support System![/bold green]",
                    border_style="green"
                ))
                break

            if user_input.lower() == 'help':
                show_help()
                continue

            if user_input.lower() == 'info':
                show_system_info(system)
                continue

            # Process the query with a nice progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("🔄 Processing your query...", total=None)
                result = await system.process_query(user_input)
                progress.update(task, completed=True)

            # Display the result
            console.print()
            format_agent_response(result)

        except KeyboardInterrupt:
            console.print(Panel(
                "[bold green]👋 Goodbye! Thank you for using the Multi-Agent Support System.[/bold green]",
                border_style="green"
            ))
            break
        except Exception as e:
            console.print(Panel(
                f"[red]❌ An unexpected error occurred: {str(e)}[/red]\n\n[yellow]Please try again or contact support if the issue persists.[/yellow]",
                title="[bold red]Error[/bold red]",
                border_style="red"
            ))


async def batch_mode(system: MultiAgentSupportSystem, queries: List[str]) -> None:
    """Run batch mode with progress tracking."""
    console.print(Panel(
        f"[bold cyan]🔄 Processing {len(queries)} queries in batch mode[/bold cyan]",
        border_style="cyan"
    ))

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        for i, query in enumerate(queries, 1):
            task = progress.add_task(f"Processing query {i}/{len(queries)}: {query[:50]}...", total=None)

            console.print(f"\n[bold yellow]📝 Query {i}/{len(queries)}:[/bold yellow] {query}")

            result = await system.process_query(query)
            results.append({"query": query, "result": result})

            progress.update(task, completed=True)
            format_agent_response(result)

    # Summary
    successful = sum(1 for r in results if r["result"]["success"])
    failed = len(results) - successful

    summary_table = Table(title="📈 Batch Processing Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="white")
    summary_table.add_column("Percentage", style="green")

    summary_table.add_row("Total Queries", str(len(results)), "100%")
    summary_table.add_row("Successful", str(successful), f"{(successful/len(results)*100):.1f}%")
    summary_table.add_row("Failed", str(failed), f"{(failed/len(results)*100):.1f}%")

    console.print()
    console.print(summary_table)


async def demo_mode(system: MultiAgentSupportSystem) -> None:
    """Run demo mode with sample queries."""
    console.print(Panel(
        "[bold cyan]🎬 Demo Mode - Sample Queries[/bold cyan]",
        border_style="cyan"
    ))

    demo_queries = [
        "How do I reset my password?",
        "My computer won't connect to the network",
        "How do I submit an expense report?",
        "What's the budget approval process?",
        "I'm having trouble with my email",
        "How do I request a new software license?"
    ]

    await batch_mode(system, demo_queries)


async def initialize_vector_stores_on_startup(system: MultiAgentSupportSystem):
    """Initialize all vector stores before starting the interactive mode."""
    rag_search_tool = system.tool_registry.get_tool('rag_search')
    if rag_search_tool and hasattr(rag_search_tool, 'rag_search'):
        await rag_search_tool.rag_search.initialize_all_vector_stores()
        console.print(Panel("[green]Vector stores initialized for all domains (if available).[/green]", border_style="green"))
    else:
        console.print(Panel("[yellow]RAG search tool not found. Skipping vector store initialization.[/yellow]", border_style="yellow"))


@app.command()
def main(
    config: Optional[str] = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    demo: bool = typer.Option(
        False,
        "--demo",
        "-d",
        help="Run in demo mode with sample queries"
    ),
    batch: Optional[List[str]] = typer.Option(
        None,
        "--batch",
        "-b",
        help="Run in batch mode with specified queries"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    🤖 Multi-Agent Support System using LangGraph and AWS Bedrock/Claude

    A sophisticated AI-powered support system that intelligently routes queries
    to specialized IT and Finance agents for comprehensive assistance.
    """

    # Check if configuration file exists
    if not Path(config).exists():
        console.print(Panel(
            f"[red]❌ Configuration file not found: {config}[/red]\n\n[yellow]Please ensure the configuration file exists and is readable.[/yellow]",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red"
        ))
        raise typer.Exit(1)

    try:
        # Initialize system with nice progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("🚀 Initializing Multi-Agent Support System...", total=None)
            system = MultiAgentSupportSystem(config)
            progress.update(task, completed=True)

        console.print(Panel(
            "[bold green]✅ System initialized successfully![/bold green]",
            border_style="green"
        ))

        # Initialize vector stores
        asyncio.run(initialize_vector_stores_on_startup(system))

        # Run in the appropriate mode
        if demo:
            asyncio.run(demo_mode(system))
        elif batch:
            asyncio.run(batch_mode(system, batch))
        else:
            asyncio.run(interactive_mode(system))

    except KeyboardInterrupt:
        console.print(Panel(
            "[bold green]👋 Goodbye![/bold green]",
            border_style="green"
        ))
        raise typer.Exit(0)
    except Exception as e:
        console.print(Panel(
            f"[red]❌ Failed to initialize system: {str(e)}[/red]\n\n[yellow]Please check your configuration and try again.[/yellow]",
            title="[bold red]Initialization Error[/bold red]",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def init_vector_store(
    domain: str = typer.Option(
        "finance",
        "--domain",
        "-d",
        help="Domain to initialize vector store for (finance or it)"
    ),
    config: Optional[str] = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """
    Manually initialize the vector store for a specific domain (finance or it).
    """
    if not Path(config).exists():
        console.print(Panel(
            f"[red]❌ Configuration file not found: {config}[/red]\n\n[yellow]Please ensure the configuration file exists and is readable.[/yellow]",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red"
        ))
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"🔄 Initializing vector store for domain: {domain}...", total=None)
        system = MultiAgentSupportSystem(config)
        rag_search_tool = system.tool_registry.get_tool('rag_search')
        if rag_search_tool and hasattr(rag_search_tool, 'rag_search'):
            asyncio.run(rag_search_tool.rag_search._ensure_domain_initialized(domain))
            console.print(Panel(f"[green]Vector store initialized for domain: {domain} (if available).[/green]", border_style="green"))
        else:
            console.print(Panel(f"[yellow]RAG search tool not found. Skipping vector store initialization for {domain}.[/yellow]", border_style="yellow"))
        progress.update(task, completed=True)


if __name__ == "__main__":
    import sys

    # If no command line arguments are provided (like when running from PyCharm),
    # run the interactive mode directly
    if len(sys.argv) == 1:
        try:
            # Initialize system with default config
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("🚀 Initializing Multi-Agent Support System...", total=None)
                system = MultiAgentSupportSystem("config.yaml")
                progress.update(task, completed=True)

            console.print(Panel(
                "[bold green]✅ System initialized successfully![/bold green]",
                border_style="green"
            ))

            # Initialize vector stores
            asyncio.run(initialize_vector_stores_on_startup(system))

            # Run interactive mode
            asyncio.run(interactive_mode(system))

        except KeyboardInterrupt:
            console.print(Panel(
                "[bold green]👋 Goodbye![/bold green]",
                border_style="green"
            ))
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Failed to initialize system: {str(e)}[/red]\n\n[yellow]Please check your configuration and try again.[/yellow]",
                title="[bold red]Initialization Error[/bold red]",
                border_style="red"
            ))
    else:
        # Run Typer CLI app for command-line usage
        app()

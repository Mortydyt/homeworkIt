"""
Day 7 - Reporting and Visualization System
Implementation of ReportBuilder with charts and data export capabilities
"""

import json
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import os

from bank_system import Bank
from transaction_system import Transaction, TransactionType, TransactionStatus
from audit_risk_system import AuditLog, RiskAnalyzer, RiskLevel
from bank_account import Currency


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    title: str
    date_range: tuple[datetime, datetime]
    include_charts: bool = True
    export_format: str = "json"  # json, csv, txt
    chart_style: str = "default"


class ReportBuilder:
    """Comprehensive reporting and visualization system"""

    def __init__(self, bank: Bank, audit_log: AuditLog, risk_analyzer: RiskAnalyzer):
        """
        Initialize report builder

        Args:
            bank: Bank system instance
            audit_log: Audit log instance
            risk_analyzer: Risk analyzer instance
        """
        self.bank = bank
        self.audit_log = audit_log
        self.risk_analyzer = risk_analyzer

        # Configure matplotlib for non-interactive use
        plt.switch_backend('Agg')
        plt.style.use('default')

        # Chart configuration
        self.chart_colors = {
            'RUB': '#2E8B57',      # SeaGreen
            'USD': '#4169E1',      # RoyalBlue
            'EUR': '#FF6347',      # Tomato
            'KZT': '#FFD700',      # Gold
            'CNY': '#DC143C',      # Crimson
            'success': '#90EE90',  # LightGreen
            'failed': '#FFB6C1',   # LightPink
            'pending': '#87CEEB'   # SkyBlue
        }

    def generate_client_report(self, client_id: str, config: ReportConfig) -> Dict:
        """Generate comprehensive client report"""
        print(f"📊 Generating client report for {client_id}...")

        client = self.bank.clients.get(client_id)
        if not client:
            return {"error": f"Client {client_id} not found"}

        # Get client accounts
        accounts = self.bank.get_client_accounts(client_id)

        # Calculate client statistics
        total_balance = 0.0
        account_breakdown = {}
        transaction_count = 0

        for account in accounts:
            balance = account.get_balance()
            currency = account.currency.value
            total_balance += balance
            account_breakdown[currency] = account_breakdown.get(currency, 0) + balance

        # Get transaction history from audit log
        client_entries = self.audit_log.get_entries(
            start_time=config.date_range[0],
            end_time=config.date_range[1],
            limit=1000
        )

        # Filter transactions for this client
        client_transactions = [entry for entry in client_entries
                             if entry.category == "TRANSACTION" and
                             (entry.account_number in [acc.account_number for acc in accounts])]

        transaction_count = len(client_transactions)

        # Get risk profile
        risk_profile = self.risk_analyzer.get_client_risk_profile(client_id)

        # Build report data
        report_data = {
            "report_title": config.title,
            "client_info": client.get_client_info(),
            "report_period": {
                "start": config.date_range[0].isoformat(),
                "end": config.date_range[1].isoformat()
            },
            "accounts": [
                {
                    "account_number": account.account_number[-4:],
                    "type": account.__class__.__name__,
                    "balance": account.get_balance(),
                    "currency": account.currency.value,
                    "status": account.status.value
                }
                for account in accounts
            ],
            "financial_summary": {
                "total_balance": total_balance,
                "account_breakdown": account_breakdown,
                "account_count": len(accounts)
            },
            "transaction_summary": {
                "total_transactions": transaction_count,
                "recent_transactions": client_transactions[:10]
            },
            "risk_analysis": risk_profile,
            "generated_at": datetime.now().isoformat()
        }

        # Generate charts if requested
        if config.include_charts:
            charts = self._generate_client_charts(report_data, client_id)
            report_data["charts"] = charts

        return report_data

    def generate_bank_report(self, config: ReportConfig) -> Dict:
        """Generate comprehensive bank report"""
        print("🏦 Generating bank-wide report...")

        # Get bank information
        bank_info = self.bank.get_bank_info()

        # Get account statistics
        all_accounts = list(self.bank.accounts.values())
        active_accounts = [acc for acc in all_accounts if acc.status.value == "active"]

        # Currency distribution
        currency_distribution = {}
        total_volume = 0.0

        for account in active_accounts:
            currency = account.currency.value
            balance = account.get_balance()
            currency_distribution[currency] = currency_distribution.get(currency, 0) + balance
            total_volume += balance

        # Get transaction statistics
        audit_entries = self.audit_log.get_entries(
            start_time=config.date_range[0],
            end_time=config.date_range[1],
            limit=5000
        )

        transaction_entries = [entry for entry in audit_entries if entry.category == "TRANSACTION"]
        status_distribution = {"completed": 0, "failed": 0, "unknown": 0}

        for entry in transaction_entries:
            if "completed" in entry.message.lower():
                status_distribution["completed"] += 1
            elif "failed" in entry.message.lower():
                status_distribution["failed"] += 1
            else:
                status_distribution["unknown"] += 1

        # Get top clients
        top_clients = self.bank.get_clients_ranking(10)

        # Get risk statistics
        risk_stats = self.risk_analyzer.get_risk_statistics()

        # Build report data
        report_data = {
            "report_title": config.title,
            "bank_info": bank_info,
            "report_period": {
                "start": config.date_range[0].isoformat(),
                "end": config.date_range[1].isoformat()
            },
            "account_statistics": {
                "total_accounts": len(all_accounts),
                "active_accounts": len(active_accounts),
                "currency_distribution": currency_distribution,
                "total_volume": total_volume
            },
            "transaction_statistics": {
                "total_transactions": len(transaction_entries),
                "status_distribution": status_distribution,
                "success_rate": (status_distribution["completed"] / len(transaction_entries) * 100
                               if len(transaction_entries) > 0 else 0)
            },
            "top_clients": top_clients,
            "risk_analysis": risk_stats,
            "audit_statistics": self.audit_log.get_statistics(),
            "generated_at": datetime.now().isoformat()
        }

        # Generate charts if requested
        if config.include_charts:
            charts = self._generate_bank_charts(report_data)
            report_data["charts"] = charts

        return report_data

    def generate_risk_report(self, config: ReportConfig) -> Dict:
        """Generate comprehensive risk analysis report"""
        print("⚠️ Generating risk analysis report...")

        # Get all risk alerts
        risk_alerts = self.risk_analyzer.risk_alerts

        # Filter by date range
        filtered_alerts = [alert for alert in risk_alerts
                          if config.date_range[0] <= alert.timestamp <= config.date_range[1]]

        # Risk level distribution
        risk_level_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        client_risk_counts = {}
        common_factors = {}

        for alert in filtered_alerts:
            if not alert.resolved:
                risk_level_distribution[alert.risk_level.value] += 1

                client_id = alert.client_id
                client_risk_counts[client_id] = client_risk_counts.get(client_id, 0) + 1

                for factor in alert.factors:
                    factor_type = factor.split(':')[0].strip()
                    common_factors[factor_type] = common_factors.get(factor_type, 0) + 1

        # Get high-risk clients
        high_risk_clients = sorted(client_risk_counts.items(),
                                 key=lambda x: x[1], reverse=True)[:10]

        # Recent audit entries related to security
        security_entries = self.audit_log.get_entries(
            category="SECURITY",
            start_time=config.date_range[0],
            end_time=config.date_range[1],
            limit=500
        )

        # Build report data
        report_data = {
            "report_title": config.title,
            "report_period": {
                "start": config.date_range[0].isoformat(),
                "end": config.date_range[1].isoformat()
            },
            "risk_overview": {
                "total_alerts": len(filtered_alerts),
                "active_alerts": len([a for a in filtered_alerts if not a.resolved]),
                "risk_level_distribution": risk_level_distribution
            },
            "high_risk_clients": [
                {"client_id": client_id, "alert_count": count}
                for client_id, count in high_risk_clients
            ],
            "common_risk_factors": common_factors,
            "recent_security_events": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level.value,
                    "message": entry.message,
                    "details": entry.details
                }
                for entry in security_entries[:20]
            ],
            "recommendations": self._generate_risk_recommendations(filtered_alerts),
            "generated_at": datetime.now().isoformat()
        }

        # Generate charts if requested
        if config.include_charts:
            charts = self._generate_risk_charts(report_data)
            report_data["charts"] = charts

        return report_data

    def _generate_client_charts(self, report_data: Dict, client_id: str) -> Dict:
        """Generate charts for client report"""
        charts = {}

        # Account balance pie chart
        if report_data["financial_summary"]["account_breakdown"]:
            chart_file = f"client_{client_id}_balance_pie.png"
            self._create_pie_chart(
                data=report_data["financial_summary"]["account_breakdown"],
                title=f"Account Balance Distribution - {client_id}",
                filename=chart_file
            )
            charts["balance_distribution"] = chart_file

        return charts

    def _generate_bank_charts(self, report_data: Dict) -> Dict:
        """Generate charts for bank report"""
        charts = {}

        # Currency distribution pie chart
        if report_data["account_statistics"]["currency_distribution"]:
            chart_file = "bank_currency_distribution.png"
            self._create_pie_chart(
                data=report_data["account_statistics"]["currency_distribution"],
                title="Bank Currency Distribution",
                filename=chart_file
            )
            charts["currency_distribution"] = chart_file

        # Transaction status pie chart
        if report_data["transaction_statistics"]["status_distribution"]:
            chart_file = "bank_transaction_status.png"
            status_data = report_data["transaction_statistics"]["status_distribution"]
            self._create_pie_chart(
                data=status_data,
                title="Transaction Status Distribution",
                filename=chart_file,
                colors=['success', 'failed', 'pending']
            )
            charts["transaction_status"] = chart_file

        # Top clients bar chart
        if report_data["top_clients"]:
            chart_file = "bank_top_clients.png"
            top_clients = report_data["top_clients"][:5]
            client_names = [client["full_name"][:15] + "..." if len(client["full_name"]) > 15 else client["full_name"]
                           for client in top_clients]
            client_balances = [client["total_balance"] for client in top_clients]

            self._create_bar_chart(
                x_data=client_names,
                y_data=client_balances,
                title="Top 5 Clients by Balance",
                x_label="Clients",
                y_label="Balance (RUB)",
                filename=chart_file
            )
            charts["top_clients"] = chart_file

        return charts

    def _generate_risk_charts(self, report_data: Dict) -> Dict:
        """Generate charts for risk report"""
        charts = {}

        # Risk level distribution pie chart
        if report_data["risk_overview"]["risk_level_distribution"]:
            chart_file = "risk_level_distribution.png"
            risk_data = report_data["risk_overview"]["risk_level_distribution"]
            self._create_pie_chart(
                data=risk_data,
                title="Risk Level Distribution",
                filename=chart_file
            )
            charts["risk_level_distribution"] = chart_file

        # Common risk factors bar chart
        if report_data["common_risk_factors"]:
            chart_file = "risk_factors.png"
            factors = list(report_data["common_risk_factors"].keys())
            counts = list(report_data["common_risk_factors"].values())

            self._create_bar_chart(
                x_data=factors,
                y_data=counts,
                title="Common Risk Factors",
                x_label="Risk Factor",
                y_label="Count",
                filename=chart_file
            )
            charts["risk_factors"] = chart_file

        return charts

    def _create_pie_chart(self, data: Dict, title: str, filename: str,
                         colors: List[str] = None) -> None:
        """Create pie chart"""
        plt.figure(figsize=(10, 8))

        # Prepare data
        labels = list(data.keys())
        sizes = list(data.values())

        # Prepare colors
        if not colors:
            colors = [self.chart_colors.get(label, '#CCCCCC') for label in labels]
        else:
            colors = [self.chart_colors.get(color, '#CCCCCC') for color in colors]

        # Create pie chart
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)

        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')

        # Add title
        plt.title(title, fontsize=14, fontweight='bold', pad=20)

        # Adjust layout
        plt.tight_layout()

        # Save chart
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  📈 Created pie chart: {filename}")

    def _create_bar_chart(self, x_data: List[str], y_data: List[float],
                         title: str, x_label: str, y_label: str,
                         filename: str) -> None:
        """Create bar chart"""
        plt.figure(figsize=(12, 8))

        # Create bar chart
        bars = plt.bar(x_data, y_data, color=self.chart_colors['RUB'], alpha=0.7)

        # Customize chart
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)

        # Rotate x-axis labels if needed
        if any(len(str(label)) > 8 for label in x_data):
            plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, y_data):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,.0f}', ha='center', va='bottom')

        # Add grid
        plt.grid(axis='y', alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save chart
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  📊 Created bar chart: {filename}")

    def export_to_json(self, report_data: Dict, filename: str) -> None:
        """Export report to JSON format"""
        json_data = json.dumps(report_data, indent=2, ensure_ascii=False)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_data)

        print(f"  💾 Exported JSON report: {filename}")

    def export_to_csv(self, report_data: Dict, filename: str) -> None:
        """Export report to CSV format (main tables only)"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow([report_data.get("report_title", "Report")])
            writer.writerow([f"Generated: {report_data.get('generated_at', '')}"])
            writer.writerow([])

            # Write different sections based on report type
            if "client_info" in report_data:
                # Client report
                writer.writerow(["Client Information"])
                client_info = report_data["client_info"]
                for key, value in client_info.items():
                    writer.writerow([key, value])
                writer.writerow([])

                if "accounts" in report_data:
                    writer.writerow(["Accounts"])
                    writer.writerow(["Type", "Balance", "Currency", "Status"])
                    for account in report_data["accounts"]:
                        writer.writerow([account["type"], account["balance"],
                                       account["currency"], account["status"]])

            elif "bank_info" in report_data:
                # Bank report
                writer.writerow(["Bank Information"])
                bank_info = report_data["bank_info"]
                for key, value in bank_info.items():
                    writer.writerow([key, value])
                writer.writerow([])

                if "top_clients" in report_data:
                    writer.writerow(["Top Clients"])
                    writer.writerow(["Rank", "Client Name", "Balance", "Account Count"])
                    for i, client in enumerate(report_data["top_clients"], 1):
                        writer.writerow([i, client["full_name"], client["total_balance"],
                                       client["account_count"]])

            elif "risk_overview" in report_data:
                # Risk report
                writer.writerow(["Risk Overview"])
                risk_overview = report_data["risk_overview"]
                for key, value in risk_overview.items():
                    writer.writerow([key, value])

        print(f"  📄 Exported CSV report: {filename}")

    def export_to_txt(self, report_data: Dict, filename: str) -> None:
        """Export report to text format"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{report_data.get('report_title', 'Report')}\n")
            f.write("=" * len(report_data.get('report_title', 'Report')) + "\n\n")
            f.write(f"Generated: {report_data.get('generated_at', '')}\n\n")

            # Write different sections based on report type
            if "client_info" in report_data:
                self._write_client_section(f, report_data)
            elif "bank_info" in report_data:
                self._write_bank_section(f, report_data)
            elif "risk_overview" in report_data:
                self._write_risk_section(f, report_data)

        print(f"  📝 Exported TXT report: {filename}")

    def _write_client_section(self, f, report_data: Dict) -> None:
        """Write client section to text file"""
        f.write("CLIENT INFORMATION\n")
        f.write("-" * 18 + "\n")
        client_info = report_data["client_info"]
        for key, value in client_info.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        f.write("\nFINANCIAL SUMMARY\n")
        f.write("-" * 18 + "\n")
        financial = report_data["financial_summary"]
        f.write(f"Total Balance: {financial['total_balance']:,.2f}\n")
        f.write(f"Account Count: {financial['account_count']}\n")

        f.write("\nAccount Breakdown:\n")
        for currency, amount in financial["account_breakdown"].items():
            f.write(f"  {currency}: {amount:,.2f}\n")

    def _write_bank_section(self, f, report_data: Dict) -> None:
        """Write bank section to text file"""
        f.write("BANK STATISTICS\n")
        f.write("-" * 16 + "\n")
        bank_info = report_data["bank_info"]
        for key, value in bank_info.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        f.write("\nTOP CLIENTS\n")
        f.write("-" * 11 + "\n")
        for i, client in enumerate(report_data["top_clients"], 1):
            f.write(f"{i}. {client['full_name']}: {client['total_balance']:,.2f} RUB\n")

    def _write_risk_section(self, f, report_data: Dict) -> None:
        """Write risk section to text file"""
        f.write("RISK OVERVIEW\n")
        f.write("-" * 13 + "\n")
        risk_overview = report_data["risk_overview"]
        for key, value in risk_overview.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        f.write("\nHIGH RISK CLIENTS\n")
        f.write("-" * 18 + "\n")
        for client in report_data["high_risk_clients"]:
            f.write(f"{client['client_id']}: {client['alert_count']} alerts\n")

    def _generate_risk_recommendations(self, alerts: List) -> List[str]:
        """Generate risk recommendations based on alerts"""
        recommendations = []

        if len(alerts) > 50:
            recommendations.append("Consider implementing stricter transaction monitoring")

        high_risk_count = len([a for a in alerts if a.risk_level == RiskLevel.HIGH])
        critical_risk_count = len([a for a in alerts if a.risk_level == RiskLevel.CRITICAL])

        if critical_risk_count > 0:
            recommendations.append(f"URGENT: {critical_risk_count} critical risk alerts require immediate attention")

        if high_risk_count > 10:
            recommendations.append(f"Review {high_risk_count} high-risk alerts for pattern analysis")

        recommendations.append("Continue regular monitoring and client education")

        return recommendations


def test_day7():
    """Test Day 7 implementation"""
    print("=== Day 7 Testing ===\n")

    # Import and setup demo system
    from demo_program import BankingSystemDemo
    demo = BankingSystemDemo()
    demo.setup_demo_data()

    # Create report builder
    report_builder = ReportBuilder(demo.bank, demo.audit_log, demo.risk_analyzer)

    # Set up report configuration
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)

    print("--- Client Report Test ---")
    client_config = ReportConfig(
        title="Client Financial Report",
        date_range=(start_time, end_time),
        include_charts=True,
        export_format="json"
    )

    client_report = report_builder.generate_client_report("CL001", client_config)
    if "error" not in client_report:
        print("✅ Client report generated successfully")
        print(f"  Total balance: {client_report['financial_summary']['total_balance']:,.2f}")
        print(f"  Account count: {client_report['financial_summary']['account_count']}")

        # Export client reports
        report_builder.export_to_json(client_report, "client_report.json")
        report_builder.export_to_csv(client_report, "client_report.csv")
        report_builder.export_to_txt(client_report, "client_report.txt")

    print("\n--- Bank Report Test ---")
    bank_config = ReportConfig(
        title="Bank Performance Report",
        date_range=(start_time, end_time),
        include_charts=True,
        export_format="json"
    )

    bank_report = report_builder.generate_bank_report(bank_config)
    print("✅ Bank report generated successfully")
    print(f"  Total accounts: {bank_report['account_statistics']['total_accounts']}")
    print(f"  Total volume: {bank_report['account_statistics']['total_volume']:,.2f}")
    print(f"  Transaction success rate: {bank_report['transaction_statistics']['success_rate']:.1f}%")

    # Export bank reports
    report_builder.export_to_json(bank_report, "bank_report.json")
    report_builder.export_to_csv(bank_report, "bank_report.csv")
    report_builder.export_to_txt(bank_report, "bank_report.txt")

    print("\n--- Risk Report Test ---")
    risk_config = ReportConfig(
        title="Risk Analysis Report",
        date_range=(start_time, end_time),
        include_charts=True,
        export_format="json"
    )

    # Create some sample transactions to generate risk data
    demo.simulate_transactions(20)

    risk_report = report_builder.generate_risk_report(risk_config)
    print("✅ Risk report generated successfully")
    print(f"  Total alerts: {risk_report['risk_overview']['total_alerts']}")
    print(f"  Active alerts: {risk_report['risk_overview']['active_alerts']}")

    # Export risk reports
    report_builder.export_to_json(risk_report, "risk_report.json")
    report_builder.export_to_csv(risk_report, "risk_report.csv")
    report_builder.export_to_txt(risk_report, "risk_report.txt")

    print("\n--- Chart Generation Summary ---")
    all_reports = [client_report, bank_report, risk_report]
    chart_count = 0
    for report in all_reports:
        if "charts" in report:
            chart_count += len(report["charts"])
            for chart_name, chart_file in report["charts"].items():
                print(f"  📈 {chart_name}: {chart_file}")

    print(f"\nTotal charts generated: {chart_count}")

    # List generated files
    print("\n--- Generated Files ---")
    import os
    for file in os.listdir('.'):
        if file.endswith(('.json', '.csv', '.txt', '.png')):
            size = os.path.getsize(file)
            print(f"  📄 {file} ({size:,} bytes)")

    print("\n=== Day 7 Tests Completed ===")


if __name__ == "__main__":
    test_day7()
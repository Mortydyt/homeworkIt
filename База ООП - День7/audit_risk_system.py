"""
Day 5 - Audit and Risk Analysis
Implementation of AuditLog and RiskAnalyzer systems
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import json
import os
from collections import defaultdict

from transaction_system import Transaction, TransactionType, TransactionStatus, TransactionPriority
from bank_account import AbstractAccount, AccountStatus, Currency


class LogLevel(Enum):
    """Log level enumeration"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """Audit log entry"""
    timestamp: datetime
    level: LogLevel
    category: str
    message: str
    details: dict
    user_id: Optional[str] = None
    account_number: Optional[str] = None
    transaction_id: Optional[str] = None
    ip_address: Optional[str] = None


@dataclass
class RiskAlert:
    """Risk alert information"""
    alert_id: str
    timestamp: datetime
    risk_level: RiskLevel
    client_id: str
    description: str
    factors: List[str]
    transaction_ids: List[str]
    recommendations: List[str]
    resolved: bool = False


class AuditLog:
    """Audit logging system"""

    def __init__(self, log_file: str = "audit.log", max_entries: int = 10000):
        """
        Initialize audit log

        Args:
            log_file: File path for persistent logging
            max_entries: Maximum entries to keep in memory
        """
        self.log_file = log_file
        self.max_entries = max_entries
        self.entries: List[AuditEntry] = []
        self.category_counts = defaultdict(int)

    def log(self, level: LogLevel, category: str, message: str,
            details: dict = None, user_id: str = None,
            account_number: str = None, transaction_id: str = None,
            ip_address: str = None) -> None:
        """Add entry to audit log"""
        entry = AuditEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            details=details or {},
            user_id=user_id,
            account_number=account_number,
            transaction_id=transaction_id,
            ip_address=ip_address
        )

        self.entries.append(entry)
        self.category_counts[category] += 1

        # Maintain memory limit
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

        # Write to file if critical or error
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._write_to_file(entry)

    def log_transaction(self, transaction: Transaction, success: bool,
                       error_message: str = None) -> None:
        """Log transaction execution"""
        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"Transaction {transaction.transaction_type.value} {success and 'completed' or 'failed'}"

        details = {
            "transaction_id": transaction.transaction_id,
            "transaction_type": transaction.transaction_type.value,
            "amount": transaction.amount,
            "currency": transaction.currency.value,
            "from_account": transaction.from_account,
            "to_account": transaction.to_account,
            "priority": transaction.priority.value,
            "processing_duration": transaction.processing_duration
        }

        if not success:
            details["error"] = error_message

        self.log(level, "TRANSACTION", message, details,
                transaction_id=transaction.transaction_id,
                account_number=transaction.from_account or transaction.to_account)

    def log_login_attempt(self, user_id: str, success: bool,
                         ip_address: str = None, reason: str = None) -> None:
        """Log login attempt"""
        level = LogLevel.INFO if success else LogLevel.WARNING
        message = f"Login attempt for user {user_id} {success and 'successful' or 'failed'}"

        details = {
            "success": success
        }

        if not success and reason:
            details["reason"] = reason

        self.log(level, "AUTHENTICATION", message, details,
                user_id=user_id, ip_address=ip_address)

    def log_account_operation(self, account_number: str, operation: str,
                            success: bool, user_id: str = None,
                            details: dict = None) -> None:
        """Log account operation"""
        level = LogLevel.INFO if success else LogLevel.WARNING
        message = f"Account {operation} {success and 'successful' or 'failed'}"

        self.log(level, "ACCOUNT", message, details or {},
                user_id=user_id, account_number=account_number)

    def log_suspicious_activity(self, client_id: str, activity: str,
                              account_numbers: List[str] = None,
                              transaction_ids: List[str] = None,
                              ip_address: str = None) -> None:
        """Log suspicious activity"""
        details = {
            "client_id": client_id,
            "activity": activity,
            "account_numbers": account_numbers or [],
            "transaction_ids": transaction_ids or []
        }

        self.log(LogLevel.WARNING, "SECURITY", f"Suspicious activity detected: {activity}",
                details, user_id=client_id, ip_address=ip_address)

    def get_entries(self, level: LogLevel = None, category: str = None,
                   start_time: datetime = None, end_time: datetime = None,
                   account_number: str = None, limit: int = 100) -> List[AuditEntry]:
        """Get filtered audit entries"""
        filtered_entries = self.entries

        if level:
            filtered_entries = [e for e in filtered_entries if e.level == level]

        if category:
            filtered_entries = [e for e in filtered_entries if e.category == category]

        if start_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]

        if end_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]

        if account_number:
            filtered_entries = [e for e in filtered_entries if e.account_number == account_number]

        # Return most recent entries
        filtered_entries.sort(key=lambda e: e.timestamp, reverse=True)
        return filtered_entries[:limit]

    def get_statistics(self) -> dict:
        """Get audit log statistics"""
        level_counts = defaultdict(int)
        recent_entries = [e for e in self.entries if e.timestamp >= datetime.now() - timedelta(hours=24)]

        for entry in self.entries:
            level_counts[entry.level.value] += 1

        recent_level_counts = defaultdict(int)
        for entry in recent_entries:
            recent_level_counts[entry.level.value] += 1

        return {
            "total_entries": len(self.entries),
            "recent_entries_24h": len(recent_entries),
            "level_distribution": dict(level_counts),
            "recent_level_distribution": dict(recent_level_counts),
            "category_distribution": dict(self.category_counts)
        }

    def _write_to_file(self, entry: AuditEntry) -> None:
        """Write entry to file"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                log_data = {
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level.value,
                    "category": entry.category,
                    "message": entry.message,
                    "details": entry.details,
                    "user_id": entry.user_id,
                    "account_number": entry.account_number,
                    "transaction_id": entry.transaction_id,
                    "ip_address": entry.ip_address
                }
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Silently ignore file write errors


class RiskAnalyzer:
    """Risk analysis system"""

    def __init__(self, audit_log: AuditLog):
        """
        Initialize risk analyzer

        Args:
            audit_log: Audit log instance for logging risk events
        """
        self.audit_log = audit_log
        self.risk_thresholds = {
            "large_amount": 100000,  # RUB equivalent
            "high_frequency": 10,    # transactions per hour
            "new_account_risk": 7,   # days
            "night_risk": True,
            "rapid_succession": 5,   # transactions in 10 minutes
            "foreign_currency_risk": 50000  # foreign currency amount
        }
        self.client_profiles: Dict[str, dict] = {}
        self.risk_alerts: List[RiskAlert] = []

    def analyze_transaction(self, transaction: Transaction, account: AbstractAccount,
                          client_transactions: List[Transaction] = None) -> List[RiskAlert]:
        """Analyze transaction for risks"""
        alerts = []
        risk_factors = []

        # Check large amount
        amount_in_rub = self._convert_to_rub(transaction.amount, transaction.currency)
        if amount_in_rub > self.risk_thresholds["large_amount"]:
            risk_factors.append(f"Large transaction amount: {amount_in_rub:.2f} RUB")

        # Check high frequency
        if client_transactions:
            recent_transactions = [t for t in client_transactions
                                 if t.created_at >= datetime.now() - timedelta(hours=1)]
            if len(recent_transactions) > self.risk_thresholds["high_frequency"]:
                risk_factors.append(f"High frequency: {len(recent_transactions)} transactions in last hour")

        # Check night time operations
        current_hour = datetime.now().hour
        if self.risk_thresholds["night_risk"] and 0 <= current_hour <= 5:
            risk_factors.append("Night time operation")

        # Check rapid succession
        very_recent_transactions = [t for t in client_transactions
                                  if t.created_at >= datetime.now() - timedelta(minutes=10)]
        if len(very_recent_transactions) > self.risk_thresholds["rapid_succession"]:
            risk_factors.append(f"Rapid succession: {len(very_recent_transactions)} transactions in 10 minutes")

        # Check new account
        account_age_days = (datetime.now() - account.created_at).days if hasattr(account, 'created_at') else 365
        if account_age_days < self.risk_thresholds["new_account_risk"]:
            risk_factors.append(f"New account: {account_age_days} days old")

        # Check foreign currency
        if transaction.currency != Currency.RUB:
            if amount_in_rub > self.risk_thresholds["foreign_currency_risk"]:
                risk_factors.append(f"Large foreign currency transaction: {transaction.amount:.2f} {transaction.currency.value}")

        # Generate alerts if risk factors found
        if risk_factors:
            client_id = self._get_client_id(account)
            alert = RiskAlert(
                alert_id=f"RISK_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{transaction.transaction_id[:8]}",
                timestamp=datetime.now(),
                risk_level=self._determine_risk_level(risk_factors),
                client_id=client_id,
                description=f"Risk detected in transaction {transaction.transaction_id[:8]}",
                factors=risk_factors,
                transaction_ids=[transaction.transaction_id],
                recommendations=self._generate_recommendations(risk_factors)
            )

            alerts.append(alert)
            self.risk_alerts.append(alert)

            # Log suspicious activity
            self.audit_log.log_suspicious_activity(
                client_id, f"Transaction risk detected: {'; '.join(risk_factors)}",
                account_numbers=[transaction.from_account, transaction.to_account],
                transaction_ids=[transaction.transaction_id]
            )

        return alerts

    def analyze_client_pattern(self, client_id: str, transactions: List[Transaction],
                             accounts: List[AbstractAccount]) -> List[RiskAlert]:
        """Analyze client behavior patterns"""
        alerts = []

        # Update client profile
        self._update_client_profile(client_id, transactions, accounts)

        # Analyze patterns
        profile = self.client_profiles.get(client_id, {})

        # Check for sudden increase in activity
        recent_24h = [t for t in transactions if t.created_at >= datetime.now() - timedelta(hours=24)]
        avg_daily_transactions = profile.get("avg_daily_transactions", 1)
        if len(recent_24h) > avg_daily_transactions * 5:
            alert = RiskAlert(
                alert_id=f"PATTERN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{client_id}",
                timestamp=datetime.now(),
                risk_level=RiskLevel.HIGH,
                client_id=client_id,
                description=f"Sudden increase in transaction activity",
                factors=[f"24h transactions: {len(recent_24h)} vs average: {avg_daily_transactions:.1f}"],
                transaction_ids=[t.transaction_id for t in recent_24h],
                recommendations=["Monitor client activity", "Consider temporary limits"]
            )
            alerts.append(alert)
            self.risk_alerts.append(alert)

        # Check for multiple new accounts
        new_accounts = [acc for acc in accounts if self._get_account_age_days(acc) < 30]
        if len(new_accounts) > 2:
            alert = RiskAlert(
                alert_id=f"NEWACC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{client_id}",
                timestamp=datetime.now(),
                risk_level=RiskLevel.MEDIUM,
                client_id=client_id,
                description=f"Multiple new accounts detected",
                factors=[f"New accounts (30 days): {len(new_accounts)}"],
                transaction_ids=[],
                recommendations=["Enhanced monitoring", "Document verification"]
            )
            alerts.append(alert)
            self.risk_alerts.append(alert)

        return alerts

    def should_block_transaction(self, transaction: Transaction, alerts: List[RiskAlert]) -> bool:
        """Determine if transaction should be blocked"""
        for alert in alerts:
            if alert.risk_level == RiskLevel.CRITICAL:
                return True

        # Check for specific high-risk patterns
        if transaction.amount > self.risk_thresholds["large_amount"] * 2:
            return True

        return False

    def get_client_risk_profile(self, client_id: str) -> dict:
        """Get risk profile for client"""
        profile = self.client_profiles.get(client_id, {})
        client_alerts = [a for a in self.risk_alerts if a.client_id == client_id and not a.resolved]

        return {
            "client_id": client_id,
            "profile": profile,
            "active_alerts": len(client_alerts),
            "risk_factors": self._get_client_risk_factors(client_id),
            "recommendations": self._get_client_recommendations(client_id)
        }

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a risk alert"""
        for alert in self.risk_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.audit_log.log(LogLevel.INFO, "RISK", f"Alert {alert_id} resolved",
                                  {"alert_id": alert_id})
                return True
        return False

    def get_risk_statistics(self) -> dict:
        """Get risk analysis statistics"""
        total_alerts = len(self.risk_alerts)
        active_alerts = len([a for a in self.risk_alerts if not a.resolved])
        risk_level_counts = defaultdict(int)
        factor_counts = defaultdict(int)

        for alert in self.risk_alerts:
            if not alert.resolved:
                risk_level_counts[alert.risk_level.value] += 1
                for factor in alert.factors:
                    # Extract main factor type
                    factor_type = factor.split(':')[0].strip()
                    factor_counts[factor_type] += 1

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "risk_level_distribution": dict(risk_level_counts),
            "common_risk_factors": dict(factor_counts),
            "monitored_clients": len(self.client_profiles)
        }

    def _determine_risk_level(self, risk_factors: List[str]) -> RiskLevel:
        """Determine risk level based on factors"""
        if len(risk_factors) >= 4 or "Large transaction amount" in risk_factors:
            return RiskLevel.CRITICAL
        elif len(risk_factors) >= 2:
            return RiskLevel.HIGH
        elif len(risk_factors) >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk factors"""
        recommendations = []

        if "Large transaction amount" in " ".join(risk_factors):
            recommendations.append("Additional verification required")
            recommendations.append("Consider transaction limits")

        if "High frequency" in " ".join(risk_factors):
            recommendations.append("Implement rate limiting")
            recommendations.append("Monitor for automated activity")

        if "Night time operation" in " ".join(risk_factors):
            recommendations.append("Enhanced monitoring during night hours")

        if "New account" in " ".join(risk_factors):
            recommendations.append("Increased monitoring for new accounts")
            recommendations.append("Verify account ownership")

        return recommendations

    def _convert_to_rub(self, amount: float, currency: Currency) -> float:
        """Convert amount to RUB for comparison"""
        exchange_rates = {
            Currency.USD: 91.5,
            Currency.EUR: 99.2,
            Currency.KZT: 0.19,
            Currency.CNY: 12.6,
            Currency.RUB: 1.0
        }
        return amount * exchange_rates.get(currency, 1.0)

    def _get_client_id(self, account: AbstractAccount) -> str:
        """Extract client ID from account (simplified)"""
        return f"CLIENT_{account.account_number[:4]}"

    def _get_account_age_days(self, account: AbstractAccount) -> int:
        """Get account age in days"""
        # Simplified - in real implementation would track creation date
        return 365  # Assume old account for testing

    def _update_client_profile(self, client_id: str, transactions: List[Transaction],
                             accounts: List[AbstractAccount]) -> None:
        """Update client risk profile"""
        if client_id not in self.client_profiles:
            self.client_profiles[client_id] = {
                "transactions_count": 0,
                "total_amount": 0.0,
                "first_transaction": None,
                "last_transaction": None,
                "avg_daily_transactions": 1.0
            }

        profile = self.client_profiles[client_id]
        profile["transactions_count"] = len(transactions)
        profile["total_amount"] = sum(t.amount for t in transactions)
        profile["account_count"] = len(accounts)

        if transactions:
            profile["first_transaction"] = min(t.created_at for t in transactions).isoformat()
            profile["last_transaction"] = max(t.created_at for t in transactions).isoformat()

            # Calculate average daily transactions
            if len(transactions) > 1:
                date_range = (max(t.created_at for t in transactions) -
                            min(t.created_at for t in transactions)).days
                if date_range > 0:
                    profile["avg_daily_transactions"] = len(transactions) / date_range

    def _get_client_risk_factors(self, client_id: str) -> List[str]:
        """Get risk factors for client"""
        client_alerts = [a for a in self.risk_alerts if a.client_id == client_id and not a.resolved]
        all_factors = []
        for alert in client_alerts:
            all_factors.extend(alert.factors)
        return list(set(all_factors))

    def _get_client_recommendations(self, client_id: str) -> List[str]:
        """Get recommendations for client"""
        client_alerts = [a for a in self.risk_alerts if a.client_id == client_id and not a.resolved]
        all_recommendations = []
        for alert in client_alerts:
            all_recommendations.extend(alert.recommendations)
        return list(set(all_recommendations))


def test_day5():
    """Test Day 5 implementation"""
    print("=== Day 5 Testing ===\n")

    # Create audit log and risk analyzer
    audit_log = AuditLog("test_audit.log")
    risk_analyzer = RiskAnalyzer(audit_log)

    print("--- Audit Log Tests ---")

    # Test different log entries
    audit_log.log_login_attempt("user123", True, "192.168.1.1")
    audit_log.log_login_attempt("user124", False, "192.168.1.2", "Invalid password")
    audit_log.log_account_operation("ACC001", "deposit", True, "user123", {"amount": 1000})
    audit_log.log_suspicious_activity("user125", "Multiple failed logins",
                                     account_numbers=["ACC001"], ip_address="192.168.1.3")

    print("Audit log entries created")

    # Test statistics
    stats = audit_log.get_statistics()
    print(f"Audit log statistics: {stats}")

    print("\n--- Risk Analysis Tests ---")

    # Create test account
    from bank_account import BankAccount
    account = BankAccount("Test User", Currency.RUB, "RISK001")
    account.deposit(50000)

    # Create risky transactions
    risky_transactions = [
        Transaction(TransactionType.TRANSFER, 150000, Currency.RUB,
                   from_account="RISK001", to_account="RISK002",  # Large amount
                   priority=TransactionPriority.HIGH),
        Transaction(TransactionType.WITHDRAWAL, 2000, Currency.USD,  # Foreign currency
                   from_account="RISK001"),
        Transaction(TransactionType.TRANSFER, 50000, Currency.RUB,
                   from_account="RISK001", to_account="RISK003",
                   priority=TransactionPriority.URGENT)
    ]

    # Analyze transactions
    all_alerts = []
    for i, transaction in enumerate(risky_transactions):
        # Simulate time progression
        if i > 0:
            transaction.created_at = datetime.now() - timedelta(minutes=i*5)

        alerts = risk_analyzer.analyze_transaction(transaction, account, risky_transactions[:i])
        all_alerts.extend(alerts)

        print(f"Transaction {i+1}: {len(alerts)} risk alerts")
        for alert in alerts:
            print(f"  - {alert.risk_level.value.upper()}: {alert.description}")

    # Test client pattern analysis
    client_id = "CLIENT_RISK"
    client_alerts = risk_analyzer.analyze_client_pattern(client_id, risky_transactions, [account])
    all_alerts.extend(client_alerts)

    print(f"\nClient pattern analysis: {len(client_alerts)} additional alerts")

    # Test transaction blocking
    print("\n--- Transaction Blocking Tests ---")
    for i, transaction in enumerate(risky_transactions):
        alerts = [a for a in all_alerts if transaction.transaction_id in a.transaction_ids]
        should_block = risk_analyzer.should_block_transaction(transaction, alerts)
        print(f"Transaction {i+1} should block: {should_block}")

    # Test client risk profile
    print("\n--- Client Risk Profile ---")
    risk_profile = risk_analyzer.get_client_risk_profile(client_id)
    print(f"Client risk profile:")
    print(f"  Active alerts: {risk_profile['active_alerts']}")
    print(f"  Risk factors: {risk_profile['risk_factors']}")
    print(f"  Recommendations: {risk_profile['recommendations']}")

    # Test risk statistics
    print("\n--- Risk Statistics ---")
    risk_stats = risk_analyzer.get_risk_statistics()
    print(f"Risk analysis statistics: {risk_stats}")

    # Test alert resolution
    print("\n--- Alert Resolution ---")
    if all_alerts:
        resolved = risk_analyzer.resolve_alert(all_alerts[0].alert_id)
        print(f"Resolved first alert: {resolved}")

        updated_stats = risk_analyzer.get_risk_statistics()
        print(f"Updated active alerts: {updated_stats['active_alerts']}")

    # Test audit log filtering
    print("\n--- Audit Log Filtering ---")
    warning_entries = audit_log.get_entries(level=LogLevel.WARNING, limit=5)
    print(f"Recent warning entries: {len(warning_entries)}")
    for entry in warning_entries:
        print(f"  - {entry.category}: {entry.message}")

    print("\n=== Day 5 Tests Completed ===")


if __name__ == "__main__":
    test_day5()
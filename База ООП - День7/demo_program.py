"""
Day 6 - Demonstration Program
Comprehensive demonstration of the entire banking system
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict
import time

from bank_system import Bank, AuthenticationResult
from transaction_system import TransactionProcessor, TransactionQueue, Transaction, TransactionType, TransactionPriority
from audit_risk_system import AuditLog, RiskAnalyzer
from bank_account import BankAccount, Currency
from advanced_accounts import SavingsAccount, PremiumAccount, InvestmentAccount, PortfolioType


class BankingSystemDemo:
    """Comprehensive banking system demonstration"""

    def __init__(self):
        """Initialize demo system"""
        print("🏦 Initializing Banking System Demo...")

        # Initialize core components
        self.bank = Bank("ТехноБанк", "DEMO001")
        self.audit_log = AuditLog("demo_audit.log")
        self.risk_analyzer = RiskAnalyzer(self.audit_log)
        self.transaction_queue = TransactionQueue()
        self.accounts = {}

        # Demo statistics
        self.stats = {
            "clients_created": 0,
            "accounts_created": 0,
            "transactions_processed": 0,
            "transactions_failed": 0,
            "risk_alerts": 0
        }

    def setup_demo_data(self) -> None:
        """Set up demo clients and accounts"""
        print("\n📋 Setting up demo data...")

        # Create demo clients
        clients_data = [
            ("CL001", "Иван Петров", 28, "+7(916)123-45-67", "ivan.petrov@email.com", "password123"),
            ("CL002", "Мария Иванова", 35, "+7(916)234-56-78", "maria.ivanova@email.com", "securepass"),
            ("CL003", "Алексей Сидоров", 42, "+7(916)345-67-89", "alexey.sidorov@email.com", "mypass"),
            ("CL004", "Елена Козлова", 31, "+7(916)456-78-90", "elena.kozlova@email.com", "newpass"),
            ("CL005", "Дмитрий Новиков", 39, "+7(916)567-89-01", "dmitry.novikov@email.com", "strongpass"),
            ("CL006", "Ольга Белова", 26, "+7(916)678-90-12", "olga.belova@email.com", "mypassword"),
            ("CL007", "Сергей Морозов", 45, "+7(916)789-01-23", "sergey.morozov@email.com", "pass123"),
            ("CL008", "Наталья Волкова", 33, "+7(916)890-12-34", "natalya.volkova@email.com", "newpass123"),
        ]

        for client_data in clients_data:
            try:
                client = self.bank.add_client(*client_data)
                self.stats["clients_created"] += 1
                print(f"  ✅ Created client: {client.full_name}")
            except Exception as e:
                print(f"  ❌ Failed to create client {client_data[1]}: {e}")

        # Create diverse accounts
        account_configs = [
            ("CL001", "bank", Currency.RUB, {}),
            ("CL001", "savings", Currency.RUB, {"min_balance": 5000, "monthly_rate": 0.04}),
            ("CL002", "premium", Currency.USD, {"overdraft_limit": 3000, "monthly_fee": 25}),
            ("CL003", "investment", Currency.EUR, {}),
            ("CL003", "bank", Currency.RUB, {}),
            ("CL004", "savings", Currency.RUB, {"min_balance": 10000, "monthly_rate": 0.05}),
            ("CL005", "premium", Currency.RUB, {"overdraft_limit": 10000, "monthly_fee": 100}),
            ("CL006", "bank", Currency.USD, {}),
            ("CL007", "investment", Currency.RUB, {}),
            ("CL008", "savings", Currency.USD, {"min_balance": 2000, "monthly_rate": 0.03}),
        ]

        for client_id, account_type, currency, kwargs in account_configs:
            try:
                account = self.bank.open_account(client_id, account_type, currency, **kwargs)
                self.accounts[account.account_number] = account
                self.stats["accounts_created"] += 1
                print(f"  ✅ Created {account_type} account: {account.account_number}")
            except Exception as e:
                print(f"  ❌ Failed to create {account_type} account for {client_id}: {e}")

        # Initialize accounts with some balance
        initial_balances = [
            ("CL001", 15000),
            ("CL002", 8000),
            ("CL003", 20000),
            ("CL004", 12000),
            ("CL005", 25000),
            ("CL006", 5000),
            ("CL007", 18000),
            ("CL008", 7000),
        ]

        for client_id, balance in initial_balances:
            accounts = self.bank.get_client_accounts(client_id)
            if accounts:
                account = random.choice(accounts)
                account.deposit(balance)
                print(f"  💰 Deposited {balance} {account.currency.value} to {account.account_number}")

        print(f"\n📊 Setup complete: {self.stats['clients_created']} clients, {self.stats['accounts_created']} accounts")

    def simulate_transactions(self, count: int = 40) -> None:
        """Simulate various transactions"""
        print(f"\n🔄 Simulating {count} transactions...")

        account_numbers = list(self.accounts.keys())

        if len(account_numbers) < 2:
            print("❌ Not enough accounts for transaction simulation")
            return

        transaction_types = [
            TransactionType.DEPOSIT,
            TransactionType.WITHDRAWAL,
            TransactionType.TRANSFER,
            TransactionType.FEE
        ]

        transactions_created = 0
        failed_transactions = 0

        for i in range(count):
            try:
                # Create random transaction
                tx_type = random.choice(transaction_types)
                amount = random.uniform(100, 50000)
                currency = random.choice(list(Currency))

                transaction = None

                if tx_type == TransactionType.DEPOSIT:
                    to_account = random.choice(account_numbers)
                    transaction = Transaction(tx_type, amount, currency, to_account=to_account)

                elif tx_type == TransactionType.WITHDRAWAL:
                    from_account = random.choice(account_numbers)
                    transaction = Transaction(tx_type, amount, currency, from_account=from_account)

                elif tx_type == TransactionType.TRANSFER:
                    from_account = random.choice(account_numbers)
                    to_account = random.choice([acc for acc in account_numbers if acc != from_account])
                    priority = random.choice(list(TransactionPriority))
                    transaction = Transaction(
                        tx_type, amount, currency,
                        from_account=from_account,
                        to_account=to_account,
                        priority=priority
                    )

                elif tx_type == TransactionType.FEE:
                    from_account = random.choice(account_numbers)
                    transaction = Transaction(tx_type, amount, currency, from_account=from_account,
                                             description="Monthly maintenance fee")

                if transaction:
                    # Add transaction without delays for demo
                    self.transaction_queue.add_transaction(transaction)
                    transactions_created += 1

            except Exception as e:
                failed_transactions += 1
                print(f"  ❌ Failed to create transaction {i+1}: {e}")

        print(f"  📝 Created {transactions_created} transactions")

        # Process transactions
        time.sleep(0.2)  # Small delay to allow scheduled transactions
        self._process_transaction_queue()

    def _process_transaction_queue(self) -> None:
        """Process all transactions in the queue"""
        print("  ⚙️ Processing transactions...")

        processor = TransactionProcessor(self.accounts)
        processed_count = 0

        # Process in batches to simulate real system
        for batch in range(10):  # 10 batches
            batch_processed = 0

            # Get scheduled transactions that are ready
            ready_scheduled = self.transaction_queue.get_scheduled_transactions()
            for transaction in ready_scheduled:
                self.transaction_queue.add_transaction(transaction)

            # Process transactions in this batch
            for _ in range(5):  # Process up to 5 transactions per batch
                transaction = self.transaction_queue.get_next_transaction()
                if not transaction:
                    break

                print(f"    🔸 Processing: {transaction.transaction_type.value} {transaction.amount:.2f} {transaction.currency.value}")

                # Perform risk analysis
                from_account = self.accounts.get(transaction.from_account) if transaction.from_account else None
                if from_account:
                    alerts = self.risk_analyzer.analyze_transaction(transaction, from_account, [])
                    self.stats["risk_alerts"] += len(alerts)

                    # Check if transaction should be blocked
                    if self.risk_analyzer.should_block_transaction(transaction, alerts):
                        print(f"      🚫 Transaction BLOCKED due to high risk")
                        self.audit_log.log_transaction(transaction, False, "Blocked by risk analysis")
                        self.stats["transactions_failed"] += 1
                        continue

                # Process transaction
                success = processor.process_transaction(transaction)
                self.audit_log.log_transaction(transaction, success)

                if success:
                    processed_count += 1
                    print(f"      ✅ Completed")
                else:
                    self.stats["transactions_failed"] += 1
                    print(f"      ❌ Failed: {transaction.failure_reason}")

                batch_processed += 1

            if batch_processed == 0:
                break

        self.stats["transactions_processed"] = processed_count
        print(f"  ✅ Processed {processed_count} transactions")

    def demonstrate_client_operations(self) -> None:
        """Demonstrate client-specific operations"""
        print("\n👤 Demonstrating Client Operations...")

        # Client authentication
        print("  🔐 Testing authentication:")
        auth_result = self.bank.authenticate_client("CL001", "password123")
        print(f"    CL001 correct password: {auth_result.value}")

        auth_result = self.bank.authenticate_client("CL002", "wrongpassword")
        print(f"    CL002 wrong password: {auth_result.value}")

        # Client accounts and balances
        print("\n  💳 Client accounts and balances:")
        for client_id in ["CL001", "CL003", "CL005"]:
            client = self.bank.clients.get(client_id)
            if client:
                print(f"    {client.full_name} ({client_id}):")
                accounts = self.bank.get_client_accounts(client_id)
                for account in accounts:
                    print(f"      {account.__class__.__name__} {account.account_number[-4:]}: "
                          f"{account.get_balance():.2f} {account.currency.value}")

        # Demonstrate investment operations
        print("\n  📈 Investment operations:")
        investment_accounts = [acc for acc in self.accounts.values()
                             if isinstance(acc, InvestmentAccount)]
        if investment_accounts:
            # Find an investment account with funds or add funds
            inv_account = None
            for acc in investment_accounts:
                if acc.get_balance() > 0:
                    inv_account = acc
                    break

            if not inv_account and investment_accounts:
                inv_account = investment_accounts[0]
                inv_account.deposit(15000)  # Add funds for demonstration

            if inv_account:
                print(f"    Investment account: {inv_account.account_number[-4:]}")
                print(f"    Available balance: {inv_account.get_balance():.2f} {inv_account.currency.value}")

                # Add investments
                try:
                    inv_account.invest(PortfolioType.STOCKS, 5000)
                    inv_account.invest(PortfolioType.BONDS, 3000)
                    inv_account.invest(PortfolioType.ETF, 2000)

                    print(f"      Total invested: {inv_account.get_total_investment_value():.2f} {inv_account.currency.value}")

                    # Project growth
                    growth_rates = {
                        PortfolioType.STOCKS: 0.12,
                        PortfolioType.BONDS: 0.05,
                        PortfolioType.ETF: 0.08
                    }
                    yearly_growth = inv_account.project_yearly_growth(growth_rates)
                    print(f"      Projected yearly growth: {yearly_growth:.2f} {inv_account.currency.value}")
                except Exception as e:
                    print(f"      ❌ Investment operation failed: {e}")
        else:
            print("    No investment accounts found")

        # Demonstrate savings account interest
        print("\n  🏦 Savings account interest:")
        savings_accounts = [acc for acc in self.accounts.values()
                          if isinstance(acc, SavingsAccount)]
        if savings_accounts:
            sav_account = savings_accounts[0]
            original_balance = sav_account.get_balance()
            interest = sav_account.apply_monthly_interest()
            print(f"    Account {sav_account.account_number[-4:]}:")
            print(f"      Balance before: {original_balance:.2f} {sav_account.currency.value}")
            print(f"      Monthly interest: {interest:.2f} {sav_account.currency.value}")
            print(f"      Balance after: {sav_account.get_balance():.2f} {sav_account.currency.value}")

    def demonstrate_risk_and_audit(self) -> None:
        """Demonstrate risk analysis and audit capabilities"""
        print("\n🔍 Demonstrating Risk Analysis and Audit...")

        # Risk statistics
        risk_stats = self.risk_analyzer.get_risk_statistics()
        print("  📊 Risk Analysis Statistics:")
        print(f"    Total alerts: {risk_stats['total_alerts']}")
        print(f"    Active alerts: {risk_stats['active_alerts']}")
        print(f"    Risk level distribution: {risk_stats['risk_level_distribution']}")
        print(f"    Common risk factors: {risk_stats['common_risk_factors']}")

        # Client risk profiles
        print("\n  👥 Client Risk Profiles:")
        for client_id in ["CL001", "CL002", "CL003"]:
            risk_profile = self.risk_analyzer.get_client_risk_profile(client_id)
            print(f"    {client_id}:")
            print(f"      Active alerts: {risk_profile['active_alerts']}")
            if risk_profile['risk_factors']:
                print(f"      Risk factors: {risk_profile['risk_factors'][:2]}")  # Show first 2

        # Audit log statistics
        audit_stats = self.audit_log.get_statistics()
        print("\n  📋 Audit Log Statistics:")
        print(f"    Total entries: {audit_stats['total_entries']}")
        print(f"    Recent entries (24h): {audit_stats['recent_entries_24h']}")
        print(f"    Level distribution: {audit_stats['level_distribution']}")
        print(f"    Category distribution: {audit_stats['category_distribution']}")

        # Recent audit entries
        print("\n  📝 Recent Critical/Warning Entries:")
        recent_entries = self.audit_log.get_entries(limit=5)
        for entry in recent_entries:
            if entry.level.value in ['CRITICAL', 'ERROR', 'WARNING']:
                print(f"    {entry.level.value}: {entry.category} - {entry.message}")

    def generate_final_report(self) -> None:
        """Generate final demonstration report"""
        print("\n📊 Final Demonstration Report")
        print("=" * 50)

        # Bank statistics
        bank_info = self.bank.get_bank_info()
        print(f"\n🏦 Bank Statistics:")
        print(f"  Total clients: {bank_info['total_clients']}")
        print(f"  Active clients: {bank_info['active_clients']}")
        print(f"  Total accounts: {bank_info['total_accounts']}")
        print(f"  Active accounts: {bank_info['active_accounts']}")

        # Transaction statistics
        print(f"\n💳 Transaction Statistics:")
        print(f"  Transactions processed: {self.stats['transactions_processed']}")
        print(f"  Transactions failed: {self.stats['transactions_failed']}")
        success_rate = (self.stats['transactions_processed'] /
                       (self.stats['transactions_processed'] + self.stats['transactions_failed']) * 100
                       if (self.stats['transactions_processed'] + self.stats['transactions_failed']) > 0 else 0)
        print(f"  Success rate: {success_rate:.1f}%")

        # Top clients by balance
        print(f"\n🏆 Top 3 Clients by Balance:")
        top_clients = self.bank.get_clients_ranking(3)
        for i, client in enumerate(top_clients, 1):
            print(f"  {i}. {client['full_name']}: {client['total_balance']:.2f} RUB "
                  f"({client['account_count']} accounts)")

        # Risk and security
        risk_stats = self.risk_analyzer.get_risk_statistics()
        print(f"\n🔍 Risk & Security:")
        print(f"  Risk alerts generated: {self.stats['risk_alerts']}")
        print(f"  Active risk alerts: {risk_stats['active_alerts']}")
        print(f"  Clients monitored: {risk_stats['monitored_clients']}")

        # Total bank balance by currency
        total_balances = self.bank.get_total_balance()
        print(f"\n💰 Total Bank Balance by Currency:")
        for currency, amount in total_balances.items():
            print(f"  {currency.value}: {amount:,.2f}")

        print("\n✨ Demonstration completed successfully!")


def main():
    """Main demonstration function"""
    print("🚀 Starting Comprehensive Banking System Demonstration")
    print("=" * 60)

    # Initialize and run demo
    demo = BankingSystemDemo()

    # Step 1: Setup demo data
    demo.setup_demo_data()

    # Step 2: Simulate transactions
    demo.simulate_transactions(35)  # Simulate 35 transactions

    # Step 3: Demonstrate client operations
    demo.demonstrate_client_operations()

    # Step 4: Demonstrate risk and audit capabilities
    demo.demonstrate_risk_and_audit()

    # Step 5: Generate final report
    demo.generate_final_report()


if __name__ == "__main__":
    main()
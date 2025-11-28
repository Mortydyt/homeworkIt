"""
Day 3 - Bank System
Implementation of Client and Bank classes with security features
"""

from datetime import datetime, time
from typing import List, Dict, Optional, Union
from enum import Enum
import re

from bank_account import AbstractAccount, AccountStatus, Currency, BankAccount
from advanced_accounts import SavingsAccount, PremiumAccount, InvestmentAccount


class ClientStatus(Enum):
    """Client status enumeration"""
    ACTIVE = "active"
    BLOCKED = "blocked"
    PENDING = "pending"


class AuthenticationResult(Enum):
    """Authentication result enumeration"""
    SUCCESS = "success"
    INVALID_CREDENTIALS = "invalid_credentials"
    ACCOUNT_BLOCKED = "account_blocked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    NIGHT_TIME_RESTRICTION = "night_time_restriction"


class Client:
    """Client of the bank"""

    def __init__(self, client_id: str, full_name: str, age: int,
                 phone: str, email: str, password: str):
        """
        Initialize client

        Args:
            client_id: Unique client identifier
            full_name: Full name of the client
            age: Age of the client (must be >= 18)
            phone: Phone number
            email: Email address
            password: Client password for authentication
        """
        if age < 18:
            raise ValueError("Client must be at least 18 years old")

        if not self._validate_email(email):
            raise ValueError("Invalid email format")

        if not self._validate_phone(phone):
            raise ValueError("Invalid phone format")

        self.client_id = client_id
        self.full_name = full_name
        self.age = age
        self.phone = phone
        self.email = email
        self._password = password
        self.status = ClientStatus.ACTIVE
        self.account_numbers: List[str] = []
        self.failed_login_attempts = 0
        self.suspicious_activities: List[str] = []
        self.last_login: Optional[datetime] = None

    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def _validate_phone(self, phone: str) -> bool:
        """Validate phone format (simple validation)"""
        pattern = r'^\+?[\d\s\-\(\)]{10,}$'
        return re.match(pattern, phone) is not None

    def add_account(self, account_number: str) -> None:
        """Add account number to client's account list"""
        if account_number not in self.account_numbers:
            self.account_numbers.append(account_number)

    def remove_account(self, account_number: str) -> None:
        """Remove account number from client's account list"""
        if account_number in self.account_numbers:
            self.account_numbers.remove(account_number)

    def verify_password(self, password: str) -> bool:
        """Verify client password"""
        return self._password == password

    def record_failed_login(self) -> None:
        """Record failed login attempt"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 3:
            self.status = ClientStatus.BLOCKED
            self.suspicious_activities.append(f"Account blocked due to {self.failed_login_attempts} failed login attempts")

    def reset_failed_attempts(self) -> None:
        """Reset failed login attempts"""
        self.failed_login_attempts = 0

    def add_suspicious_activity(self, activity: str) -> None:
        """Add suspicious activity record"""
        self.suspicious_activities.append(f"{datetime.now().isoformat()}: {activity}")

    def get_client_info(self) -> dict:
        """Get client information"""
        return {
            "client_id": self.client_id,
            "full_name": self.full_name,
            "age": self.age,
            "phone": self.phone,
            "email": self.email,
            "status": self.status.value,
            "account_count": len(self.account_numbers),
            "account_numbers": self.account_numbers.copy(),
            "failed_login_attempts": self.failed_login_attempts,
            "suspicious_activities_count": len(self.suspicious_activities),
            "last_login": self.last_login.isoformat() if self.last_login else None
        }

    def __str__(self) -> str:
        """String representation of client"""
        return (f"Client[{self.client_id}] {self.full_name} - "
                f"{self.status.value.upper()} - {len(self.account_numbers)} accounts")


class Bank:
    """Main bank system class"""

    def __init__(self, bank_name: str, bank_code: str):
        """
        Initialize bank

        Args:
            bank_name: Name of the bank
            bank_code: Bank identification code
        """
        self.bank_name = bank_name
        self.bank_code = bank_code
        self.clients: Dict[str, Client] = {}
        self.accounts: Dict[str, AbstractAccount] = {}
        self.suspicious_operations: List[Dict] = []

    def add_client(self, client_id: str, full_name: str, age: int,
                   phone: str, email: str, password: str) -> Client:
        """Add new client to the bank"""
        if client_id in self.clients:
            raise ValueError(f"Client with ID {client_id} already exists")

        client = Client(client_id, full_name, age, phone, email, password)
        self.clients[client_id] = client
        return client

    def authenticate_client(self, client_id: str, password: str) -> AuthenticationResult:
        """Authenticate client with security checks"""
        if client_id not in self.clients:
            return AuthenticationResult.INVALID_CREDENTIALS

        client = self.clients[client_id]

        # Check if client is blocked
        if client.status == ClientStatus.BLOCKED:
            return AuthenticationResult.ACCOUNT_BLOCKED

        # Check night time restriction (00:00 - 05:00)
        current_time = datetime.now().time()
        night_start = time(0, 0)
        night_end = time(5, 0)
        if night_start <= current_time <= night_end:
            return AuthenticationResult.NIGHT_TIME_RESTRICTION

        # Verify password
        if client.verify_password(password):
            client.reset_failed_attempts()
            client.last_login = datetime.now()
            return AuthenticationResult.SUCCESS
        else:
            client.record_failed_login()
            client.add_suspicious_activity("Failed login attempt")
            return AuthenticationResult.INVALID_CREDENTIALS

    def open_account(self, client_id: str, account_type: str,
                    currency: Currency = Currency.RUB, **kwargs) -> AbstractAccount:
        """Open new account for client"""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found")

        client = self.clients[client_id]

        # Create account based on type
        if account_type.lower() == "bank":
            account = BankAccount(client.full_name, currency)
        elif account_type.lower() == "savings":
            account = SavingsAccount(client.full_name, currency, **kwargs)
        elif account_type.lower() == "premium":
            account = PremiumAccount(client.full_name, currency, **kwargs)
        elif account_type.lower() == "investment":
            account = InvestmentAccount(client.full_name, currency, **kwargs)
        else:
            raise ValueError(f"Unknown account type: {account_type}")

        # Register account
        self.accounts[account.account_number] = account
        client.add_account(account.account_number)

        return account

    def close_account(self, account_number: str) -> bool:
        """Close account"""
        if account_number not in self.accounts:
            return False

        account = self.accounts[account_number]
        account.close_account()
        return True

    def freeze_account(self, account_number: str, reason: str = "") -> bool:
        """Freeze account"""
        if account_number not in self.accounts:
            return False

        account = self.accounts[account_number]
        account.freeze_account()

        # Log suspicious activity
        if reason:
            client_id = self._find_client_by_account(account_number)
            if client_id:
                self.clients[client_id].add_suspicious_activity(f"Account {account_number} frozen: {reason}")

        return True

    def unfreeze_account(self, account_number: str) -> bool:
        """Unfreeze account"""
        if account_number not in self.accounts:
            return False

        account = self.accounts[account_number]
        account.unfreeze_account()
        return True

    def search_accounts(self, client_id: Optional[str] = None,
                       account_type: Optional[str] = None,
                       currency: Optional[Currency] = None) -> List[AbstractAccount]:
        """Search accounts with filters"""
        results = []

        for account in self.accounts.values():
            # Filter by client
            if client_id:
                account_client_id = self._find_client_by_account(account.account_number)
                if account_client_id != client_id:
                    continue

            # Filter by type
            if account_type:
                account_type_name = account.__class__.__name__.lower()
                if account_type.lower() not in account_type_name:
                    continue

            # Filter by currency
            if currency and account.currency != currency:
                continue

            results.append(account)

        return results

    def get_client_accounts(self, client_id: str) -> List[AbstractAccount]:
        """Get all accounts for a specific client"""
        if client_id not in self.clients:
            return []

        client = self.clients[client_id]
        accounts = []
        for account_number in client.account_numbers:
            if account_number in self.accounts:
                accounts.append(self.accounts[account_number])

        return accounts

    def get_total_balance(self) -> Dict[Currency, float]:
        """Get total balance for all accounts by currency"""
        total_by_currency = {}

        for account in self.accounts.values():
            if account.status == AccountStatus.ACTIVE:
                currency = account.currency
                balance = account.get_balance()

                if currency not in total_by_currency:
                    total_by_currency[currency] = 0.0
                total_by_currency[currency] += balance

        return total_by_currency

    def get_clients_ranking(self, limit: int = 10) -> List[Dict]:
        """Get ranking of clients by total balance"""
        client_balances = []

        for client in self.clients.values():
            accounts = self.get_client_accounts(client.client_id)
            total_balance = sum(acc.get_balance() for acc in accounts
                              if acc.status == AccountStatus.ACTIVE and acc.currency == Currency.RUB)

            client_balances.append({
                "client_id": client.client_id,
                "full_name": client.full_name,
                "total_balance": total_balance,
                "account_count": len(accounts)
            })

        # Sort by total balance (descending)
        client_balances.sort(key=lambda x: x["total_balance"], reverse=True)

        return client_balances[:limit]

    def _find_client_by_account(self, account_number: str) -> Optional[str]:
        """Find client ID by account number"""
        for client_id, client in self.clients.items():
            if account_number in client.account_numbers:
                return client_id
        return None

    def get_bank_info(self) -> dict:
        """Get bank information"""
        active_accounts = sum(1 for acc in self.accounts.values()
                             if acc.status == AccountStatus.ACTIVE)
        active_clients = sum(1 for client in self.clients.values()
                           if client.status == ClientStatus.ACTIVE)

        return {
            "bank_name": self.bank_name,
            "bank_code": self.bank_code,
            "total_clients": len(self.clients),
            "active_clients": active_clients,
            "total_accounts": len(self.accounts),
            "active_accounts": active_accounts,
            "suspicious_operations_count": len(self.suspicious_operations)
        }


def test_day3():
    """Test Day 3 implementation"""
    print("=== Day 3 Testing ===\n")

    # Create bank
    bank = Bank("ТехноБанк", "TBANK001")
    print(f"Created bank: {bank.bank_name}")

    # Add clients
    client1 = bank.add_client("CL001", "Иван Петров", 25, "+7(999)123-45-67", "ivan@example.com", "password123")
    client2 = bank.add_client("CL002", "Мария Иванова", 32, "+7(999)987-65-43", "maria@example.com", "securepass")

    print(f"Added clients: {client1}, {client2}")

    # Test authentication
    print("\n--- Authentication Tests ---")
    auth_result = bank.authenticate_client("CL001", "password123")
    print(f"Client 1 auth: {auth_result.value}")

    auth_result = bank.authenticate_client("CL002", "wrongpass")
    print(f"Client 2 wrong password: {auth_result.value}")

    # Open accounts
    print("\n--- Account Management ---")
    account1 = bank.open_account("CL001", "bank", Currency.RUB)
    account2 = bank.open_account("CL001", "savings", Currency.RUB, min_balance=5000, monthly_rate=0.04)
    account3 = bank.open_account("CL002", "premium", Currency.USD, overdraft_limit=3000)

    print(f"Opened accounts for client 1: {account1.account_number}, {account2.account_number}")
    print(f"Opened account for client 2: {account3.account_number}")

    # Deposit some money
    account1.deposit(10000)
    account2.deposit(15000)
    account3.deposit(2000)

    # Test account operations
    print(f"\nClient 1 accounts: {len(bank.get_client_accounts('CL001'))}")
    print(f"Client 2 accounts: {len(bank.get_client_accounts('CL002'))}")

    # Test search
    print("\n--- Search Tests ---")
    all_accounts = bank.search_accounts()
    print(f"Total accounts in bank: {len(all_accounts)}")

    savings_accounts = bank.search_accounts(account_type="savings")
    print(f"Savings accounts: {len(savings_accounts)}")

    usd_accounts = bank.search_accounts(currency=Currency.USD)
    print(f"USD accounts: {len(usd_accounts)}")

    # Test freezing
    print("\n--- Security Tests ---")
    bank.freeze_account(account1.account_number, "Suspicious activity detected")
    print(f"Froze account {account1.account_number}")

    # Test total balance
    total_balances = bank.get_total_balance()
    print(f"\nTotal bank balance by currency: {total_balances}")

    # Test clients ranking
    ranking = bank.get_clients_ranking()
    print(f"\nTop clients by balance:")
    for i, client in enumerate(ranking, 1):
        print(f"{i}. {client['full_name']}: {client['total_balance']:.2f} RUB")

    # Bank info
    bank_info = bank.get_bank_info()
    print(f"\nBank info: {bank_info}")

    print("\n=== Day 3 Tests Completed ===")


if __name__ == "__main__":
    test_day3()
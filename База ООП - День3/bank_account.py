"""
Day 1 - Basic Banking Account Model
Implementation of abstract account and basic bank account with custom exceptions
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import uuid


class AccountStatus(Enum):
    """Account status enumeration"""
    ACTIVE = "active"
    FROZEN = "frozen"
    CLOSED = "closed"


class Currency(Enum):
    """Supported currencies"""
    RUB = "RUB"
    USD = "USD"
    EUR = "EUR"
    KZT = "KZT"
    CNY = "CNY"


# Custom Exceptions
class AccountFrozenError(Exception):
    """Raised when attempting to operate on a frozen account"""
    pass


class AccountClosedError(Exception):
    """Raised when attempting to operate on a closed account"""
    pass


class InvalidOperationError(Exception):
    """Raised for invalid operations"""
    pass


class InsufficientFundsError(Exception):
    """Raised when insufficient funds for withdrawal"""
    pass


class AbstractAccount(ABC):
    """Abstract base class for all bank accounts"""

    def __init__(self, owner_name: str, currency: Currency = Currency.RUB, account_number: Optional[str] = None):
        """
        Initialize abstract account

        Args:
            owner_name: Name of the account owner
            currency: Account currency
            account_number: Optional custom account number
        """
        self.account_number = account_number or self._generate_short_uuid()
        self.owner_name = owner_name
        self._balance = 0.0
        self.status = AccountStatus.ACTIVE
        self.currency = currency

    def _generate_short_uuid(self) -> str:
        """Generate short UUID for account number"""
        return str(uuid.uuid4())[:8].upper()

    @abstractmethod
    def deposit(self, amount: float) -> None:
        """Deposit money into account"""
        pass

    @abstractmethod
    def withdraw(self, amount: float) -> None:
        """Withdraw money from account"""
        pass

    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account information"""
        pass

    def _validate_amount(self, amount: float) -> None:
        """Validate amount for operations"""
        if amount <= 0:
            raise InvalidOperationError("Amount must be positive")

    def _check_account_status(self) -> None:
        """Check if account allows operations"""
        if self.status == AccountStatus.CLOSED:
            raise AccountClosedError(f"Account {self.account_number} is closed")
        elif self.status == AccountStatus.FROZEN:
            raise AccountFrozenError(f"Account {self.account_number} is frozen")

    def get_balance(self) -> float:
        """Get current balance"""
        return self._balance

    def freeze_account(self) -> None:
        """Freeze account"""
        if self.status != AccountStatus.CLOSED:
            self.status = AccountStatus.FROZEN

    def unfreeze_account(self) -> None:
        """Unfreeze account"""
        if self.status == AccountStatus.FROZEN:
            self.status = AccountStatus.ACTIVE

    def close_account(self) -> None:
        """Close account"""
        self.status = AccountStatus.CLOSED


class BankAccount(AbstractAccount):
    """Basic bank account implementation"""

    def __init__(self, owner_name: str, currency: Currency = Currency.RUB, account_number: Optional[str] = None):
        super().__init__(owner_name, currency, account_number)

    def deposit(self, amount: float) -> None:
        """Deposit money into account"""
        self._validate_amount(amount)
        self._check_account_status()

        self._balance += amount

    def withdraw(self, amount: float) -> None:
        """Withdraw money from account"""
        self._validate_amount(amount)
        self._check_account_status()

        if amount > self._balance:
            raise InsufficientFundsError(f"Insufficient funds: balance={self._balance}, requested={amount}")

        self._balance -= amount

    def get_account_info(self) -> dict:
        """Get account information"""
        return {
            "account_number": self.account_number,
            "owner_name": self.owner_name,
            "balance": self._balance,
            "currency": self.currency.value,
            "status": self.status.value,
            "account_type": "BankAccount"
        }

    def __str__(self) -> str:
        """String representation of account"""
        return (f"BankAccount[{self.account_number[-4:]}] - {self.owner_name} - "
                f"{self.status.value.upper()} - {self._balance:.2f} {self.currency.value}")


def test_day1():
    """Test Day 1 implementation"""
    print("=== Day 1 Testing ===\n")

    # Create active account
    account1 = BankAccount("Иван Петров", Currency.RUB)
    print(f"Created: {account1}")

    # Test deposit
    account1.deposit(1000.0)
    print(f"After deposit: {account1}")

    # Test withdrawal
    account1.withdraw(300.0)
    print(f"After withdrawal: {account1}")

    # Create frozen account
    account2 = BankAccount("Мария Иванова", Currency.USD)
    account2.deposit(500.0)
    account2.freeze_account()
    print(f"\nCreated frozen: {account2}")

    # Test operations on frozen account
    try:
        account2.withdraw(100.0)
    except AccountFrozenError as e:
        print(f"Expected error: {e}")

    # Test insufficient funds
    try:
        account1.withdraw(1000.0)
    except InsufficientFundsError as e:
        print(f"Expected error: {e}")

    # Test invalid amount
    try:
        account1.deposit(-50.0)
    except InvalidOperationError as e:
        print(f"Expected error: {e}")

    # Test account info
    print(f"\nAccount info: {account1.get_account_info()}")

    print("\n=== Day 1 Tests Completed ===")


if __name__ == "__main__":
    test_day1()
"""
Day 2 - Advanced Bank Account Types
Implementation of SavingsAccount, PremiumAccount, and InvestmentAccount
"""

from abc import abstractmethod
from enum import Enum
from typing import List, Dict, Optional
import uuid
from bank_account import AbstractAccount, AccountStatus, Currency, AccountFrozenError, AccountClosedError, InvalidOperationError, InsufficientFundsError


class PortfolioType(Enum):
    """Investment portfolio types"""
    STOCKS = "stocks"
    BONDS = "bonds"
    ETF = "etf"


class SavingsAccount(AbstractAccount):
    """Savings account with minimum balance and monthly interest"""

    def __init__(self, owner_name: str, currency: Currency = Currency.RUB,
                 min_balance: float = 1000.0, monthly_rate: float = 0.03,
                 account_number: Optional[str] = None):
        super().__init__(owner_name, currency, account_number)
        self.min_balance = min_balance
        self.monthly_rate = monthly_rate

    def deposit(self, amount: float) -> None:
        """Deposit money into savings account"""
        self._validate_amount(amount)
        self._check_account_status()
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        """Withdraw money from savings account"""
        self._validate_amount(amount)
        self._check_account_status()

        if self._balance - amount < self.min_balance:
            raise InsufficientFundsError(
                f"Cannot withdraw: would breach minimum balance of {self.min_balance} {self.currency.value}"
            )

        self._balance -= amount

    def apply_monthly_interest(self) -> float:
        """Apply monthly interest and return interest amount"""
        if self.status != AccountStatus.ACTIVE:
            return 0.0

        interest_amount = self._balance * self.monthly_rate
        self._balance += interest_amount
        return interest_amount

    def get_account_info(self) -> dict:
        """Get savings account information"""
        info = {
            "account_number": self.account_number,
            "owner_name": self.owner_name,
            "balance": self._balance,
            "currency": self.currency.value,
            "status": self.status.value,
            "account_type": "SavingsAccount",
            "min_balance": self.min_balance,
            "monthly_rate": self.monthly_rate,
            "monthly_interest": self._balance * self.monthly_rate
        }
        return info

    def __str__(self) -> str:
        """String representation of savings account"""
        return (f"SavingsAccount[{self.account_number[-4:]}] - {self.owner_name} - "
                f"{self.status.value.upper()} - {self._balance:.2f} {self.currency.value} "
                f"(min: {self.min_balance:.2f})")


class PremiumAccount(AbstractAccount):
    """Premium account with overdraft and higher limits"""

    def __init__(self, owner_name: str, currency: Currency = Currency.RUB,
                 overdraft_limit: float = 5000.0, monthly_fee: float = 50.0,
                 account_number: Optional[str] = None):
        super().__init__(owner_name, currency, account_number)
        self.overdraft_limit = overdraft_limit
        self.monthly_fee = monthly_fee
        self._fee_applied_this_month = False

    def deposit(self, amount: float) -> None:
        """Deposit money into premium account"""
        self._validate_amount(amount)
        self._check_account_status()
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        """Withdraw money from premium account (supports overdraft)"""
        self._validate_amount(amount)
        self._check_account_status()

        if amount > self._balance + self.overdraft_limit:
            raise InsufficientFundsError(
                f"Cannot withdraw: amount exceeds balance and overdraft limit. "
                f"Available: {self._balance + self.overdraft_limit:.2f}, requested: {amount:.2f}"
            )

        self._balance -= amount

    def apply_monthly_fee(self) -> None:
        """Apply monthly fee if not already applied"""
        if not self._fee_applied_this_month and self.status == AccountStatus.ACTIVE:
            self._balance -= self.monthly_fee
            self._fee_applied_this_month = True

    def reset_monthly_fee(self) -> None:
        """Reset monthly fee status for new month"""
        self._fee_applied_this_month = False

    def get_overdraft_available(self) -> float:
        """Get available overdraft amount"""
        return max(0, self.overdraft_limit + self._balance)

    def get_account_info(self) -> dict:
        """Get premium account information"""
        info = {
            "account_number": self.account_number,
            "owner_name": self.owner_name,
            "balance": self._balance,
            "currency": self.currency.value,
            "status": self.status.value,
            "account_type": "PremiumAccount",
            "overdraft_limit": self.overdraft_limit,
            "monthly_fee": self.monthly_fee,
            "overdraft_available": self.get_overdraft_available(),
            "fee_applied": self._fee_applied_this_month
        }
        return info

    def __str__(self) -> str:
        """String representation of premium account"""
        overdraft_info = f"(OD: {self.get_overdraft_available():.2f})" if self._balance < 0 else ""
        return (f"PremiumAccount[{self.account_number[-4:]}] - {self.owner_name} - "
                f"{self.status.value.upper()} - {self._balance:.2f} {self.currency.value} {overdraft_info}")


class InvestmentAccount(AbstractAccount):
    """Investment account with portfolio management"""

    def __init__(self, owner_name: str, currency: Currency = Currency.RUB,
                 account_number: Optional[str] = None):
        super().__init__(owner_name, currency, account_number)
        self.portfolio: Dict[PortfolioType, float] = {
            PortfolioType.STOCKS: 0.0,
            PortfolioType.BONDS: 0.0,
            PortfolioType.ETF: 0.0
        }

    def deposit(self, amount: float) -> None:
        """Deposit money into investment account"""
        self._validate_amount(amount)
        self._check_account_status()
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        """Withdraw money from investment account"""
        self._validate_amount(amount)
        self._check_account_status()

        if amount > self._balance:
            raise InsufficientFundsError(f"Insufficient funds: balance={self._balance}, requested={amount}")

        self._balance -= amount

    def invest(self, portfolio_type: PortfolioType, amount: float) -> None:
        """Invest money in specific portfolio type"""
        self._validate_amount(amount)
        self._check_account_status()

        if amount > self._balance:
            raise InsufficientFundsError(f"Insufficient funds for investment: balance={self._balance}, requested={amount}")

        self._balance -= amount
        self.portfolio[portfolio_type] += amount

    def sell_investment(self, portfolio_type: PortfolioType, amount: float) -> None:
        """Sell investment from specific portfolio type"""
        self._validate_amount(amount)

        if self.portfolio[portfolio_type] < amount:
            raise InsufficientFundsError(
                f"Insufficient investment in {portfolio_type.value}: "
                f"available={self.portfolio[portfolio_type]}, requested={amount}"
            )

        self.portfolio[portfolio_type] -= amount
        self._balance += amount

    def get_total_investment_value(self) -> float:
        """Get total value of investments"""
        return sum(self.portfolio.values())

    def project_yearly_growth(self, growth_rates: Dict[PortfolioType, float]) -> float:
        """Project yearly growth based on growth rates"""
        projected_growth = 0.0
        for portfolio_type, amount in self.portfolio.items():
            if amount > 0 and portfolio_type in growth_rates:
                projected_growth += amount * growth_rates[portfolio_type]
        return projected_growth

    def get_account_info(self) -> dict:
        """Get investment account information"""
        info = {
            "account_number": self.account_number,
            "owner_name": self.owner_name,
            "balance": self._balance,
            "currency": self.currency.value,
            "status": self.status.value,
            "account_type": "InvestmentAccount",
            "portfolio": {k.value: v for k, v in self.portfolio.items()},
            "total_investment_value": self.get_total_investment_value()
        }
        return info

    def __str__(self) -> str:
        """String representation of investment account"""
        total_invested = self.get_total_investment_value()
        return (f"InvestmentAccount[{self.account_number[-4:]}] - {self.owner_name} - "
                f"{self.status.value.upper()} - Cash: {self._balance:.2f} {self.currency.value}, "
                f"Invested: {total_invested:.2f} {self.currency.value}")


def test_day2():
    """Test Day 2 implementation"""
    print("=== Day 2 Testing ===\n")

    # Test Savings Account
    print("--- Savings Account ---")
    savings = SavingsAccount("Анна Смирнова", Currency.RUB, min_balance=1000.0, monthly_rate=0.05)
    savings.deposit(5000.0)
    print(f"Created: {savings}")

    interest = savings.apply_monthly_interest()
    print(f"Applied monthly interest: {interest:.2f} RUB")
    print(f"After interest: {savings}")

    try:
        savings.withdraw(4500.0)  # Should fail - below min balance
    except InsufficientFundsError as e:
        print(f"Expected error: {e}")

    # Test Premium Account
    print("\n--- Premium Account ---")
    premium = PremiumAccount("Петр Волков", Currency.USD, overdraft_limit=2000.0, monthly_fee=30.0)
    premium.deposit(1000.0)
    print(f"Created: {premium}")

    premium.withdraw(1200.0)  # Should use overdraft
    print(f"After overdraft: {premium}")
    print(f"Available overdraft: {premium.get_overdraft_available():.2f} USD")

    premium.apply_monthly_fee()
    print(f"After monthly fee: {premium}")

    # Test Investment Account
    print("\n--- Investment Account ---")
    investment = InvestmentAccount("Елена Козлова", Currency.EUR)
    investment.deposit(10000.0)
    print(f"Created: {investment}")

    investment.invest(PortfolioType.STOCKS, 3000.0)
    investment.invest(PortfolioType.BONDS, 2000.0)
    investment.invest(PortfolioType.ETF, 1500.0)
    print(f"After investments: {investment}")

    investment.sell_investment(PortfolioType.STOCKS, 500.0)
    print(f"After selling stocks: {investment}")

    # Test yearly growth projection
    growth_rates = {
        PortfolioType.STOCKS: 0.10,
        PortfolioType.BONDS: 0.04,
        PortfolioType.ETF: 0.07
    }
    yearly_growth = investment.project_yearly_growth(growth_rates)
    print(f"Projected yearly growth: {yearly_growth:.2f} EUR")

    # Print detailed account info
    print(f"\nInvestment account info: {investment.get_account_info()}")

    print("\n=== Day 2 Tests Completed ===")


if __name__ == "__main__":
    test_day2()
"""
Day 4 - Transaction System
Implementation of Transaction, TransactionQueue, and TransactionProcessor
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import uuid
import time

from bank_account import AbstractAccount, AccountStatus, Currency, AccountFrozenError, AccountClosedError, InsufficientFundsError
from advanced_accounts import PremiumAccount


class TransactionType(Enum):
    """Transaction type enumeration"""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    FEE = "fee"
    INTEREST = "interest"


class TransactionStatus(Enum):
    """Transaction status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransactionPriority(Enum):
    """Transaction priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class CurrencyRate:
    """Currency exchange rate"""
    from_currency: Currency
    to_currency: Currency
    rate: float
    timestamp: datetime


class Transaction:
    """Transaction model"""

    def __init__(self, transaction_type: TransactionType, amount: float,
                 currency: Currency, from_account: Optional[str] = None,
                 to_account: Optional[str] = None, priority: TransactionPriority = TransactionPriority.NORMAL,
                 fee: float = 0.0, description: str = ""):
        """
        Initialize transaction

        Args:
            transaction_type: Type of transaction
            amount: Transaction amount
            currency: Transaction currency
            from_account: Source account number
            to_account: Destination account number
            priority: Transaction priority
            fee: Transaction fee
            description: Transaction description
        """
        self.transaction_id = str(uuid.uuid4())
        self.transaction_type = transaction_type
        self.amount = amount
        self.currency = currency
        self.from_account = from_account
        self.to_account = to_account
        self.priority = priority
        self.fee = fee
        self.description = description
        self.status = TransactionStatus.PENDING
        self.created_at = datetime.now()
        self.processed_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.failure_reason = ""
        self.processing_duration: Optional[float] = None

    def start_processing(self) -> None:
        """Mark transaction as processing"""
        self.status = TransactionStatus.PROCESSING
        self.processed_at = datetime.now()

    def complete(self) -> None:
        """Mark transaction as completed"""
        self.status = TransactionStatus.COMPLETED
        self.completed_at = datetime.now()
        if self.processed_at:
            self.processing_duration = (self.completed_at - self.processed_at).total_seconds()

    def fail(self, reason: str) -> None:
        """Mark transaction as failed"""
        self.status = TransactionStatus.FAILED
        self.failure_reason = reason
        self.completed_at = datetime.now()
        if self.processed_at:
            self.processing_duration = (self.completed_at - self.processed_at).total_seconds()

    def cancel(self, reason: str = "") -> None:
        """Cancel transaction"""
        self.status = TransactionStatus.CANCELLED
        self.failure_reason = reason
        self.completed_at = datetime.now()

    def get_total_amount(self) -> float:
        """Get total amount including fee"""
        return self.amount + self.fee

    def get_transaction_info(self) -> dict:
        """Get transaction information"""
        return {
            "transaction_id": self.transaction_id,
            "type": self.transaction_type.value,
            "amount": self.amount,
            "currency": self.currency.value,
            "fee": self.fee,
            "total_amount": self.get_total_amount(),
            "from_account": self.from_account,
            "to_account": self.to_account,
            "priority": self.priority.value,
            "status": self.status.value,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_duration": self.processing_duration,
            "failure_reason": self.failure_reason
        }

    def __str__(self) -> str:
        """String representation of transaction"""
        return (f"{self.transaction_type.value.upper()}[{self.transaction_id[:8]}] - "
                f"{self.amount:.2f} {self.currency.value} - {self.status.value.upper()}")


class TransactionQueue:
    """Transaction queue management"""

    def __init__(self):
        """Initialize transaction queue"""
        self._queue: List[Transaction] = []
        self._scheduled_transactions: List[tuple] = []  # (datetime, transaction)
        self.max_retry_attempts = 3

    def add_transaction(self, transaction: Transaction, delay_seconds: int = 0) -> None:
        """Add transaction to queue"""
        if delay_seconds > 0:
            execute_at = datetime.now() + timedelta(seconds=delay_seconds)
            self._scheduled_transactions.append((execute_at, transaction))
        else:
            self._queue.append(transaction)

    def get_next_transaction(self) -> Optional[Transaction]:
        """Get next transaction from queue (by priority)"""
        if not self._queue:
            return None

        # Sort by priority (higher first), then by creation time (earlier first)
        self._queue.sort(key=lambda t: (t.priority.value, t.created_at), reverse=True)
        return self._queue.pop(0)

    def get_pending_count(self) -> int:
        """Get count of pending transactions"""
        return len(self._queue)

    def get_scheduled_transactions(self) -> List[Transaction]:
        """Get transactions ready to be executed from scheduled list"""
        now = datetime.now()
        ready_transactions = []
        remaining_scheduled = []

        for execute_at, transaction in self._scheduled_transactions:
            if execute_at <= now:
                ready_transactions.append(transaction)
            else:
                remaining_scheduled.append((execute_at, transaction))

        self._scheduled_transactions = remaining_scheduled
        return ready_transactions

    def cancel_transaction(self, transaction_id: str, reason: str = "") -> bool:
        """Cancel transaction by ID"""
        # Check main queue
        for i, transaction in enumerate(self._queue):
            if transaction.transaction_id == transaction_id:
                transaction.cancel(reason)
                self._queue.pop(i)
                return True

        # Check scheduled transactions
        for i, (_, transaction) in enumerate(self._scheduled_transactions):
            if transaction.transaction_id == transaction_id:
                transaction.cancel(reason)
                self._scheduled_transactions.pop(i)
                return True

        return False

    def get_queue_info(self) -> dict:
        """Get queue information"""
        priority_counts = {}
        for transaction in self._queue:
            priority_name = transaction.priority.name
            priority_counts[priority_name] = priority_counts.get(priority_name, 0) + 1

        return {
            "pending_count": len(self._queue),
            "scheduled_count": len(self._scheduled_transactions),
            "priority_distribution": priority_counts
        }


class TransactionProcessor:
    """Transaction processing engine"""

    def __init__(self, accounts: Dict[str, AbstractAccount]):
        """
        Initialize transaction processor

        Args:
            accounts: Dictionary of account_number -> Account
        """
        self.accounts = accounts
        self.exchange_rates = {
            (Currency.USD, Currency.RUB): 91.50,
            (Currency.EUR, Currency.RUB): 99.20,
            (Currency.USD, Currency.EUR): 0.92,
            (Currency.KZT, Currency.RUB): 0.19,
            (Currency.CNY, Currency.RUB): 12.60
        }
        self.external_transfer_fee_rate = 0.02  # 2%
        self.retry_delay = 1.0  # seconds

    def process_transaction(self, transaction: Transaction) -> bool:
        """Process a single transaction"""
        transaction.start_processing()

        try:
            if transaction.transaction_type == TransactionType.DEPOSIT:
                return self._process_deposit(transaction)
            elif transaction.transaction_type == TransactionType.WITHDRAWAL:
                return self._process_withdrawal(transaction)
            elif transaction.transaction_type == TransactionType.TRANSFER:
                return self._process_transfer(transaction)
            elif transaction.transaction_type == TransactionType.FEE:
                return self._process_fee(transaction)
            elif transaction.transaction_type == TransactionType.INTEREST:
                return self._process_interest(transaction)
            else:
                transaction.fail(f"Unknown transaction type: {transaction.transaction_type}")
                return False

        except Exception as e:
            transaction.fail(f"Processing error: {str(e)}")
            return False

    def _process_deposit(self, transaction: Transaction) -> bool:
        """Process deposit transaction"""
        if not transaction.to_account or transaction.to_account not in self.accounts:
            transaction.fail("Destination account not found")
            return False

        account = self.accounts[transaction.to_account]
        account.deposit(transaction.amount)
        transaction.complete()
        return True

    def _process_withdrawal(self, transaction: Transaction) -> bool:
        """Process withdrawal transaction"""
        if not transaction.from_account or transaction.from_account not in self.accounts:
            transaction.fail("Source account not found")
            return False

        account = self.accounts[transaction.from_account]
        account.withdraw(transaction.amount)
        transaction.complete()
        return True

    def _process_transfer(self, transaction: Transaction) -> bool:
        """Process transfer transaction"""
        if not transaction.from_account or transaction.from_account not in self.accounts:
            transaction.fail("Source account not found")
            return False

        if not transaction.to_account or transaction.to_account not in self.accounts:
            transaction.fail("Destination account not found")
            return False

        from_account = self.accounts[transaction.from_account]
        to_account = self.accounts[transaction.to_account]

        # Check account status
        if from_account.status != AccountStatus.ACTIVE:
            transaction.fail("Source account is not active")
            return False

        if to_account.status != AccountStatus.ACTIVE:
            transaction.fail("Destination account is not active")
            return False

        # Currency conversion if needed
        amount_to_withdraw = transaction.amount
        if from_account.currency != transaction.currency:
            amount_to_withdraw = self._convert_currency(
                transaction.currency, from_account.currency, transaction.amount
            )

        amount_to_deposit = transaction.amount
        if to_account.currency != transaction.currency:
            amount_to_deposit = self._convert_currency(
                transaction.currency, to_account.currency, transaction.amount
            )

        # Check for sufficient funds (except premium accounts)
        if not isinstance(from_account, PremiumAccount):
            if amount_to_withdraw > from_account.get_balance():
                transaction.fail("Insufficient funds")
                return False

        # Apply external transfer fee if transferring to different client
        fee = 0.0
        if self._is_external_transfer(transaction.from_account, transaction.to_account):
            fee = amount_to_withdraw * self.external_transfer_fee_rate
            amount_to_withdraw += fee

        # Execute transfer
        try:
            from_account.withdraw(amount_to_withdraw)
            to_account.deposit(amount_to_deposit)
            transaction.complete()
            return True
        except (InsufficientFundsError, AccountFrozenError, AccountClosedError) as e:
            transaction.fail(f"Transfer failed: {str(e)}")
            return False

    def _process_fee(self, transaction: Transaction) -> bool:
        """Process fee transaction"""
        if not transaction.from_account or transaction.from_account not in self.accounts:
            transaction.fail("Source account not found")
            return False

        account = self.accounts[transaction.from_account]
        account.withdraw(transaction.amount)
        transaction.complete()
        return True

    def _process_interest(self, transaction: Transaction) -> bool:
        """Process interest transaction"""
        if not transaction.to_account or transaction.to_account not in self.accounts:
            transaction.fail("Destination account not found")
            return False

        account = self.accounts[transaction.to_account]
        account.deposit(transaction.amount)
        transaction.complete()
        return True

    def _convert_currency(self, from_currency: Currency, to_currency: Currency, amount: float) -> float:
        """Convert currency amount"""
        if from_currency == to_currency:
            return amount

        key = (from_currency, to_currency)
        if key in self.exchange_rates:
            return amount * self.exchange_rates[key]

        # Try reverse conversion
        reverse_key = (to_currency, from_currency)
        if reverse_key in self.exchange_rates:
            return amount / self.exchange_rates[reverse_key]

        # Default to 1:1 if rate not found
        return amount

    def _is_external_transfer(self, from_account: str, to_account: str) -> bool:
        """Check if transfer is external (between different clients)"""
        # This is a simplified check - in real implementation, you'd track account ownership
        return from_account[:4] != to_account[:4]

    def set_exchange_rate(self, from_currency: Currency, to_currency: Currency, rate: float) -> None:
        """Set exchange rate"""
        self.exchange_rates[(from_currency, to_currency)] = rate

    def get_processing_stats(self) -> dict:
        """Get processing statistics"""
        return {
            "exchange_rates_count": len(self.exchange_rates),
            "external_transfer_fee_rate": self.external_transfer_fee_rate,
            "accounts_managed": len(self.accounts)
        }


def test_day4():
    """Test Day 4 implementation"""
    print("=== Day 4 Testing ===\n")

    # Create accounts for testing
    accounts = {}
    from bank_account import BankAccount
    account1 = BankAccount("Alice", Currency.RUB, "ACC001")
    account2 = BankAccount("Bob", Currency.USD, "ACC002")
    account3 = PremiumAccount("Charlie", Currency.RUB, account_number="ACC003", overdraft_limit=5000)

    # Add initial balances
    account1.deposit(10000)
    account2.deposit(5000)
    account3.deposit(3000)

    accounts.update({
        "ACC001": account1,
        "ACC002": account2,
        "ACC003": account3
    })

    # Create transaction system
    queue = TransactionQueue()
    processor = TransactionProcessor(accounts)

    print("--- Transaction Queue Tests ---")

    # Create transactions
    transactions = [
        Transaction(TransactionType.DEPOSIT, 1000, Currency.RUB, to_account="ACC001"),
        Transaction(TransactionType.WITHDRAWAL, 500, Currency.RUB, from_account="ACC001"),
        Transaction(TransactionType.TRANSFER, 200, Currency.USD, from_account="ACC002", to_account="ACC003",
                   priority=TransactionPriority.HIGH),
        Transaction(TransactionType.FEE, 50, Currency.RUB, from_account="ACC003", description="Monthly fee"),
        Transaction(TransactionType.TRANSFER, 1000, Currency.RUB, from_account="ACC001", to_account="ACC003",
                   priority=TransactionPriority.URGENT),
    ]

    # Add transactions to queue (no delays for testing)
    for transaction in transactions:
        queue.add_transaction(transaction)

    print(f"Added {len(transactions)} transactions to queue")
    print(f"Queue info: {queue.get_queue_info()}")

    print("\n--- Transaction Processing Tests ---")

    # Process transactions
    processed_count = 0
    max_iterations = 20  # Prevent infinite loop
    iteration = 0

    while (queue.get_pending_count() > 0 or queue.get_scheduled_transactions()) and iteration < max_iterations:
        iteration += 1

        # Get scheduled transactions that are ready
        ready_scheduled = queue.get_scheduled_transactions()
        for transaction in ready_scheduled:
            queue.add_transaction(transaction)

        # Process next transaction
        transaction = queue.get_next_transaction()
        if transaction:
            print(f"Processing: {transaction}")
            success = processor.process_transaction(transaction)
            print(f"Result: {'SUCCESS' if success else 'FAILED'} - {transaction.status.value}")
            if transaction.status == TransactionStatus.FAILED:
                print(f"  Failure reason: {transaction.failure_reason}")
            processed_count += 1
        else:
            # Wait a bit for scheduled transactions
            time.sleep(0.5)

    # Process any remaining scheduled transactions (force them)
    remaining_scheduled = queue.get_scheduled_transactions()
    print(f"\nProcessing {len(remaining_scheduled)} remaining scheduled transactions...")
    for transaction in remaining_scheduled:
        queue.add_transaction(transaction)
        processed_tx = queue.get_next_transaction()
        if processed_tx:
            print(f"Processing: {processed_tx}")
            success = processor.process_transaction(processed_tx)
            print(f"Result: {'SUCCESS' if success else 'FAILED'} - {processed_tx.status.value}")
            if processed_tx.status == TransactionStatus.FAILED:
                print(f"  Failure reason: {processed_tx.failure_reason}")
            processed_count += 1

    print(f"\nProcessed {processed_count} transactions")

    # Test currency conversion
    print("\n--- Currency Conversion Tests ---")
    rub_to_usd = processor._convert_currency(Currency.RUB, Currency.USD, 1000)
    usd_to_rub = processor._convert_currency(Currency.USD, Currency.RUB, 100)
    print(f"1000 RUB = {rub_to_usd:.2f} USD")
    print(f"100 USD = {usd_to_rub:.2f} RUB")

    # Test account balances after processing
    print("\n--- Final Account Balances ---")
    for acc_num, account in accounts.items():
        print(f"{account}: Balance = {account.get_balance():.2f} {account.currency.value}")

    # Test transaction cancellation
    print("\n--- Transaction Cancellation Tests ---")
    cancel_transaction = Transaction(TransactionType.TRANSFER, 100, Currency.RUB,
                                   from_account="ACC001", to_account="ACC002")
    queue.add_transaction(cancel_transaction)
    cancelled = queue.cancel_transaction(cancel_transaction.transaction_id, "User requested cancellation")
    print(f"Transaction cancellation: {'SUCCESS' if cancelled else 'FAILED'}")

    # Processor stats
    print(f"\n--- Processor Stats ---")
    stats = processor.get_processing_stats()
    print(f"Stats: {stats}")

    print("\n=== Day 4 Tests Completed ===")


if __name__ == "__main__":
    test_day4()
#!/usr/bin/env python3
"""
Stock Valuation Calculator
Calculates P/E, DCF, P/S ratios and provides an investment rating grade.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Grade(Enum):
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    F = "F"


@dataclass
class StockData:
    """Input data for stock valuation."""
    # Price data
    current_price: float
    shares_outstanding: float
    
    # Earnings data
    net_income: float
    
    # Sales data
    revenue: float
    
    # DCF data
    free_cash_flow: float
    growth_rate_5y: float  # Expected growth rate for years 1-5 (as decimal)
    growth_rate_terminal: float  # Terminal growth rate (as decimal)
    discount_rate: float  # WACC / Discount rate (as decimal)
    net_debt: float = 0  # Total debt - cash (can be negative for net cash)
    annual_dividend: float = 0  # Annual dividend per share


@dataclass
class ValuationResult:
    """Results from valuation calculations."""
    pe_ratio: float
    ps_ratio: float
    dcf_value_per_share: float
    dcf_upside: float
    market_cap: float
    investment_grade: Grade
    grade_score: float
    dividend_yield: float
    payout_ratio: float


class StockValuator:
    """Main valuation calculator class."""
    
    # Grading thresholds
    PE_THRESHOLDS = {
        'excellent': 15,
        'good': 20,
        'fair': 25,
        'poor': 35
    }
    
    PS_THRESHOLDS = {
        'excellent': 2,
        'good': 4,
        'fair': 6,
        'poor': 10
    }
    
    DCF_THRESHOLDS = {
        'strong_buy': 0.30,  # 30% upside
        'buy': 0.15,
        'fair': 0,
        'overvalued': -0.15
    }
    
    def calculate_pe(self, data: StockData) -> float:
        """Calculate Price-to-Earnings ratio."""
        if data.net_income <= 0:
            return float('inf')
        eps = data.net_income / data.shares_outstanding
        return data.current_price / eps
    
    def calculate_ps(self, data: StockData) -> float:
        """Calculate Price-to-Sales ratio."""
        if data.revenue <= 0:
            return float('inf')
        market_cap = data.current_price * data.shares_outstanding
        return market_cap / data.revenue
    
    def calculate_dcf(self, data: StockData) -> float:
        """
        Calculate DCF value per share using two-stage model.
        
        Stage 1: 5 years of high growth
        Stage 2: Terminal growth perpetuity
        """
        if data.free_cash_flow <= 0:
            return 0
        
        # Validate inputs
        if data.discount_rate <= data.growth_rate_terminal:
            raise ValueError("Discount rate must be greater than terminal growth rate")
        
        # Stage 1: Project FCF for 5 years
        fcf = data.free_cash_flow
        pv_stage1 = 0
        
        for year in range(1, 6):
            fcf *= (1 + data.growth_rate_5y)
            pv_stage1 += fcf / ((1 + data.discount_rate) ** year)
        
        # Stage 2: Terminal value (Gordon Growth Model)
        terminal_fcf = fcf * (1 + data.growth_rate_terminal)
        terminal_value = terminal_fcf / (data.discount_rate - data.growth_rate_terminal)
        pv_terminal = terminal_value / ((1 + data.discount_rate) ** 5)
        
        # Enterprise Value and Equity Value
        enterprise_value = pv_stage1 + pv_terminal
        equity_value = enterprise_value - data.net_debt
        
        # Value per share
        value_per_share = equity_value / data.shares_outstanding
        
        return max(0, value_per_share)
    
    def calculate_dividend_yield(self, data: StockData) -> float:
        """Calculate dividend yield as percentage."""
        if data.current_price <= 0:
            return 0
        return (data.annual_dividend / data.current_price) * 100
    
    def calculate_payout_ratio(self, data: StockData) -> float:
        """Calculate dividend payout ratio (dividends / earnings)."""
        if data.net_income <= 0 or data.shares_outstanding <= 0:
            return 0
        eps = data.net_income / data.shares_outstanding
        if eps <= 0:
            return 0
        return (data.annual_dividend / eps) * 100
    
    def calculate_grade(self, pe: float, ps: float, dcf_upside: float) -> tuple[Grade, float]:
        """
        Calculate investment grade based on multiple metrics.
        Returns (grade, score) where score is 0-100.
        
        Grading scale:
        A+ = 90-100, A = 80-90, B+ = 70-80, B = 60-70, F = <60
        """
        score = 50  # Start neutral
        
        # P/E scoring (lower is better)
        if pe == float('inf') or pe <= 0:
            pe_score = 0
        elif pe < self.PE_THRESHOLDS['excellent']:
            pe_score = 35
        elif pe < self.PE_THRESHOLDS['good']:
            pe_score = 28
        elif pe < self.PE_THRESHOLDS['fair']:
            pe_score = 20
        elif pe < self.PE_THRESHOLDS['poor']:
            pe_score = 10
        else:
            pe_score = 5
        
        # P/S scoring (lower is better)
        if ps == float('inf') or ps <= 0:
            ps_score = 0
        elif ps < self.PS_THRESHOLDS['excellent']:
            ps_score = 30
        elif ps < self.PS_THRESHOLDS['good']:
            ps_score = 22
        elif ps < self.PS_THRESHOLDS['fair']:
            ps_score = 15
        elif ps < self.PS_THRESHOLDS['poor']:
            ps_score = 8
        else:
            ps_score = 0
        
        # DCF upside scoring (higher upside is better)
        if dcf_upside >= self.DCF_THRESHOLDS['strong_buy']:
            dcf_score = 35
        elif dcf_upside >= self.DCF_THRESHOLDS['buy']:
            dcf_score = 25
        elif dcf_upside >= self.DCF_THRESHOLDS['fair']:
            dcf_score = 15
        elif dcf_upside >= self.DCF_THRESHOLDS['overvalued']:
            dcf_score = 5
        else:
            dcf_score = 0
        
        total_score = min(100, pe_score + ps_score + dcf_score)
        
        # Convert score to letter grade (new scale)
        if total_score >= 90:
            grade = Grade.A_PLUS
        elif total_score >= 80:
            grade = Grade.A
        elif total_score >= 70:
            grade = Grade.B_PLUS
        elif total_score >= 60:
            grade = Grade.B
        else:
            grade = Grade.F
        
        return grade, total_score
    
    def valuate(self, data: StockData) -> ValuationResult:
        """Run full valuation analysis."""
        market_cap = data.current_price * data.shares_outstanding
        
        pe = self.calculate_pe(data)
        ps = self.calculate_ps(data)
        dcf_value = self.calculate_dcf(data)
        div_yield = self.calculate_dividend_yield(data)
        payout = self.calculate_payout_ratio(data)
        
        dcf_upside = (dcf_value - data.current_price) / data.current_price if data.current_price > 0 else 0
        
        grade, score = self.calculate_grade(pe, ps, dcf_upside)
        
        return ValuationResult(
            pe_ratio=pe,
            ps_ratio=ps,
            dcf_value_per_share=dcf_value,
            dcf_upside=dcf_upside,
            market_cap=market_cap,
            investment_grade=grade,
            grade_score=score,
            dividend_yield=div_yield,
            payout_ratio=payout
        )


def format_currency(value: float) -> str:
    """Format large numbers in billions/millions."""
    if abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.2f}"


def print_report(data: StockData, result: ValuationResult) -> None:
    """Print formatted valuation report."""
    print("\n" + "=" * 60)
    print("           STOCK VALUATION REPORT")
    print("=" * 60)
    
    print("\n📊 INPUT DATA:")
    print(f"  Current Price:        ${data.current_price:.2f}")
    print(f"  Shares Outstanding:   {data.shares_outstanding/1e6:.2f}M")
    print(f"  Market Cap:           {format_currency(result.market_cap)}")
    print(f"  Net Income (TTM):     {format_currency(data.net_income)}")
    print(f"  Revenue (TTM):        {format_currency(data.revenue)}")
    print(f"  Free Cash Flow:       {format_currency(data.free_cash_flow)}")
    print(f"  Net Debt:             {format_currency(data.net_debt)}")
    if data.annual_dividend > 0:
        print(f"  Annual Dividend:      ${data.annual_dividend:.2f}")
    
    print("\n📈 VALUATION METRICS:")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │ P/E Ratio:           {result.pe_ratio:>10.2f}          │")
    print(f"  │ P/S Ratio:           {result.ps_ratio:>10.2f}          │")
    print(f"  │ DCF Value/Share:     ${result.dcf_value_per_share:>9.2f}          │")
    print(f"  │ DCF Upside:          {result.dcf_upside*100:>9.1f}%          │")
    if data.annual_dividend > 0:
        print(f"  │ Dividend Yield:      {result.dividend_yield:>9.2f}%          │")
        print(f"  │ Payout Ratio:        {result.payout_ratio:>9.1f}%          │")
    print(f"  └─────────────────────────────────────────┘")
    
    print("\n🎯 INVESTMENT GRADE:")
    grade_emoji = {
        Grade.A_PLUS: "🟢", Grade.A: "🟢",
        Grade.B_PLUS: "🟡", Grade.B: "🟡",
        Grade.F: "🔴"
    }
    
    print(f"  {grade_emoji.get(result.investment_grade, '⚪')}  GRADE: {result.investment_grade.value}")
    print(f"     Score: {result.grade_score}/100")
    
    # Interpretation
    print("\n📝 INTERPRETATION:")
    if result.pe_ratio == float('inf'):
        print("  • P/E: Company not profitable (negative earnings)")
    elif result.pe_ratio < 15:
        print("  • P/E: Potentially undervalued relative to earnings")
    elif result.pe_ratio > 30:
        print("  • P/E: Potentially overvalued relative to earnings")
    else:
        print("  • P/E: Reasonable valuation relative to earnings")
    
    if result.ps_ratio < 2:
        print("  • P/S: Low price-to-sales (value territory)")
    elif result.ps_ratio > 10:
        print("  • P/S: High price-to-sales (growth premium)")
    else:
        print("  • P/S: Moderate price-to-sales")
    
    if result.dcf_upside > 0.30:
        print("  • DCF: Significantly undervalued (30%+ upside)")
    elif result.dcf_upside > 0.15:
        print("  • DCF: Moderately undervalued (15-30% upside)")
    elif result.dcf_upside > 0:
        print("  • DCF: Slightly undervalued")
    elif result.dcf_upside > -0.15:
        print("  • DCF: Fairly valued to slightly overvalued")
    else:
        print("  • DCF: Potentially overvalued")
    
    # Dividend interpretation
    if data.annual_dividend > 0:
        if result.dividend_yield >= 4:
            print(f"  • Dividend: High yield stock ({result.dividend_yield:.2f}%)")
        elif result.dividend_yield >= 2:
            print(f"  • Dividend: Moderate yield ({result.dividend_yield:.2f}%)")
        else:
            print(f"  • Dividend: Low yield ({result.dividend_yield:.2f}%)")
        
        if result.payout_ratio > 80:
            print("  • Payout: High payout ratio (may be unsustainable)")
        elif result.payout_ratio > 50:
            print("  • Payout: Moderate payout ratio")
        else:
            print("  • Payout: Conservative payout ratio (room to grow)")
    
    print("\n" + "=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Stock Valuation Calculator - P/E, DCF, P/S Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic valuation
  python stock_valuator.py --price 150 --shares 15e9 --income 100e9 --revenue 400e9 --fcf 110e9
  
  # Full DCF with growth assumptions
  python stock_valuator.py --price 150 --shares 15e9 --income 100e9 --revenue 400e9 \\
    --fcf 110e9 --growth-5y 0.10 --growth-terminal 0.03 --discount 0.08 --net-debt 50e9
        """
    )
    
    # Required arguments
    parser.add_argument('--price', '-p', type=float, required=True,
                        help='Current stock price')
    parser.add_argument('--shares', '-s', type=float, required=True,
                        help='Shares outstanding')
    parser.add_argument('--income', '-i', type=float, required=True,
                        help='Net income (TTM)')
    parser.add_argument('--revenue', '-r', type=float, required=True,
                        help='Revenue (TTM)')
    parser.add_argument('--fcf', '-f', type=float, required=True,
                        help='Free cash flow')
    
    # DCF optional arguments
    parser.add_argument('--growth-5y', '-g5', type=float, default=0.10,
                        help='Expected growth rate years 1-5 (default: 0.10 = 10%%)')
    parser.add_argument('--growth-terminal', '-gt', type=float, default=0.025,
                        help='Terminal growth rate (default: 0.025 = 2.5%%)')
    parser.add_argument('--discount', '-d', type=float, default=0.10,
                        help='Discount rate/WACC (default: 0.10 = 10%%)')
    parser.add_argument('--net-debt', '-nd', type=float, default=0,
                        help='Net debt (debt - cash, default: 0)')
    parser.add_argument('--dividend', '-div', type=float, default=0,
                        help='Annual dividend per share (default: 0)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create stock data from arguments
    data = StockData(
        current_price=args.price,
        shares_outstanding=args.shares,
        net_income=args.income,
        revenue=args.revenue,
        free_cash_flow=args.fcf,
        growth_rate_5y=args.growth_5y,
        growth_rate_terminal=args.growth_terminal,
        discount_rate=args.discount,
        net_debt=args.net_debt,
        annual_dividend=args.dividend
    )
    
    # Run valuation
    valuator = StockValuator()
    
    try:
        result = valuator.valuate(data)
        print_report(data, result)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

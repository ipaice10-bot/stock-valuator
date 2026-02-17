import streamlit as st
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
    current_price: float
    shares_outstanding: float
    net_income: float
    revenue: float
    free_cash_flow: float
    growth_rate_5y: float
    growth_rate_terminal: float
    discount_rate: float
    net_debt: float = 0
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
        'strong_buy': 0.30,
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
        """Calculate DCF value per share using two-stage model."""
        if data.free_cash_flow <= 0:
            return 0
        
        if data.discount_rate <= data.growth_rate_terminal:
            raise ValueError("Discount rate must be greater than terminal growth rate")
        
        fcf = data.free_cash_flow
        pv_stage1 = 0
        
        for year in range(1, 6):
            fcf *= (1 + data.growth_rate_5y)
            pv_stage1 += fcf / ((1 + data.discount_rate) ** year)
        
        terminal_fcf = fcf * (1 + data.growth_rate_terminal)
        terminal_value = terminal_fcf / (data.discount_rate - data.growth_rate_terminal)
        pv_terminal = terminal_value / ((1 + data.discount_rate) ** 5)
        
        enterprise_value = pv_stage1 + pv_terminal
        equity_value = enterprise_value - data.net_debt
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
        """Calculate investment grade based on multiple metrics."""
        
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
        
        # Convert score to letter grade
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


# Page config
st.set_page_config(
    page_title="Stock Valuation Calculator",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Stock Valuation Calculator")
st.markdown("Calculate P/E, DCF, P/S ratios and get an investment grade")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Basic Data")
    
    stock_name = st.text_input("Stock Name/Ticker", "AAPL")
    current_price = st.number_input("Current Stock Price ($)", min_value=0.01, value=150.0, step=1.0)
    shares_outstanding = st.number_input("Shares Outstanding", min_value=1.0, value=15.7e9, step=1e9, format="%.0f")
    
    st.subheader("💰 Financial Data (TTM)")
    net_income = st.number_input("Net Income ($)", min_value=0.0, value=97e9, step=1e9, format="%.0f")
    revenue = st.number_input("Revenue ($)", min_value=0.0, value=391e9, step=1e9, format="%.0f")
    free_cash_flow = st.number_input("Free Cash Flow ($)", min_value=0.0, value=99.8e9, step=1e9, format="%.0f")
    net_debt = st.number_input("Net Debt ($) - negative = net cash", value=0.0, step=1e9, format="%.0f")
    
    st.subheader("💵 Dividend Data")
    annual_dividend = st.number_input("Annual Dividend per Share ($)", min_value=0.0, value=0.96, step=0.01, format="%.2f")

with col2:
    st.subheader("🔮 DCF Assumptions")
    
    growth_5y = st.slider("5-Year Growth Rate (%)", min_value=-10.0, max_value=50.0, value=10.0, step=1.0) / 100
    growth_terminal = st.slider("Terminal Growth Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1) / 100
    discount_rate = st.slider("Discount Rate / WACC (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.5) / 100
    
    st.markdown("---")
    st.info("""
    **Typical Ranges:**
    - 5-Year Growth: 5-20% for most companies
    - Terminal Growth: 2-3% (near GDP growth)
    - Discount Rate: 8-12% depending on risk
    """)

# Calculate button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    calculate = st.button("🚀 Calculate Valuation", use_container_width=True, type="primary")

if calculate:
    try:
        # Create stock data
        data = StockData(
            current_price=current_price,
            shares_outstanding=shares_outstanding,
            net_income=net_income,
            revenue=revenue,
            free_cash_flow=free_cash_flow,
            growth_rate_5y=growth_5y,
            growth_rate_terminal=growth_terminal,
            discount_rate=discount_rate,
            net_debt=net_debt,
            annual_dividend=annual_dividend
        )
        
        # Run valuation
        valuator = StockValuator()
        result = valuator.valuate(data)
        
        # Results section
        st.markdown("---")
        st.header(f"📋 Valuation Report: {stock_name}")
        
        # Grade card
        grade_colors = {
            Grade.A_PLUS: "#00C853",
            Grade.A: "#64DD17",
            Grade.B_PLUS: "#FFD600",
            Grade.B: "#FFAB00",
            Grade.F: "#FF1744"
        }
        
        grade_color = grade_colors.get(result.investment_grade, "#9E9E9E")
        
        st.markdown(f"""
        <div style="background-color: {grade_color}20; border: 3px solid {grade_color}; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 30px;">
            <h2 style="margin: 0; color: {grade_color};">Investment Grade: {result.investment_grade.value}</h2>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Score: {result.grade_score}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics in columns
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("P/E Ratio", f"{result.pe_ratio:.2f}")
        with m2:
            st.metric("P/S Ratio", f"{result.ps_ratio:.2f}")
        with m3:
            st.metric("DCF Value", f"${result.dcf_value_per_share:.2f}")
        with m4:
            upside_color = "normal" if result.dcf_upside >= 0 else "inverse"
            st.metric("DCF Upside", f"{result.dcf_upside*100:.1f}%", delta_color=upside_color)
        
        # Dividend metrics row
        if result.dividend_yield > 0:
            st.subheader("💵 Dividend Metrics")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.metric("Dividend Yield", f"{result.dividend_yield:.2f}%")
            with d2:
                st.metric("Annual Dividend", f"${annual_dividend:.2f}")
            with d3:
                st.metric("Payout Ratio", f"{result.payout_ratio:.1f}%")
        
        # Detailed breakdown
        st.subheader("📊 Detailed Metrics")
        
        det_col1, det_col2 = st.columns(2)
        
        with det_col1:
            st.markdown("**Input Summary**")
            dividend_row = f"| Annual Dividend | ${annual_dividend:.2f} |\n" if annual_dividend > 0 else ""
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Current Price | ${current_price:.2f} |
            | Market Cap | {format_currency(result.market_cap)} |
            | Net Income | {format_currency(net_income)} |
            | Revenue | {format_currency(revenue)} |
            | Free Cash Flow | {format_currency(free_cash_flow)} |
            | Net Debt | {format_currency(net_debt)} |
            {dividend_row}""")
        
        with det_col2:
            st.markdown("**DCF Assumptions**")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | 5-Year Growth | {growth_5y*100:.1f}% |
            | Terminal Growth | {growth_terminal*100:.1f}% |
            | Discount Rate | {discount_rate*100:.1f}% |
            """)
        
        # Interpretation
        st.subheader("📝 Interpretation")
        
        interpretations = []
        
        if result.pe_ratio == float('inf'):
            interpretations.append("• **P/E**: Company not profitable (negative earnings)")
        elif result.pe_ratio < 15:
            interpretations.append("• **P/E**: Potentially undervalued relative to earnings")
        elif result.pe_ratio > 30:
            interpretations.append("• **P/E**: Potentially overvalued relative to earnings")
        else:
            interpretations.append("• **P/E**: Reasonable valuation relative to earnings")
        
        if result.ps_ratio < 2:
            interpretations.append("• **P/S**: Low price-to-sales (value territory)")
        elif result.ps_ratio > 10:
            interpretations.append("• **P/S**: High price-to-sales (growth premium)")
        else:
            interpretations.append("• **P/S**: Moderate price-to-sales")
        
        # Dividend interpretation
        if result.dividend_yield > 0:
            if result.dividend_yield >= 4:
                interpretations.append(f"• **Dividend**: High yield stock ({result.dividend_yield:.2f}%)")
            elif result.dividend_yield >= 2:
                interpretations.append(f"• **Dividend**: Moderate yield ({result.dividend_yield:.2f}%)")
            else:
                interpretations.append(f"• **Dividend**: Low yield ({result.dividend_yield:.2f}%)")
            
            if result.payout_ratio > 80:
                interpretations.append("• **Payout**: High payout ratio (may be unsustainable)")
            elif result.payout_ratio > 50:
                interpretations.append("• **Payout**: Moderate payout ratio")
            else:
                interpretations.append("• **Payout**: Conservative payout ratio (room to grow)")
        
        if result.dcf_upside > 0.30:
            interpretations.append("• **DCF**: Significantly undervalued (30%+ upside)")
        elif result.dcf_upside > 0.15:
            interpretations.append("• **DCF**: Moderately undervalued (15-30% upside)")
        elif result.dcf_upside > 0:
            interpretations.append("• **DCF**: Slightly undervalued")
        elif result.dcf_upside > -0.15:
            interpretations.append("• **DCF**: Fairly valued to slightly overvalued")
        else:
            interpretations.append("• **DCF**: Potentially overvalued")
        
        for interp in interpretations:
            st.markdown(interp)
        
        # Grade explanation
        st.markdown("---")
        st.subheader("🎯 Grading Scale")
        st.markdown("""
        | Grade | Score Range |
        |-------|-------------|
        | **A+** | 90-100 |
        | **A** | 80-89 |
        | **B+** | 70-79 |
        | **B** | 60-69 |
        | **F** | <60 |
        """)
        
    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

else:
    # Show instructions when not calculating
    st.markdown("---")
    st.info("""
    ### How to Use
    1. Enter the stock's basic financial data on the left
    2. Adjust DCF assumptions on the right
    3. Click **Calculate Valuation** to see the results
    
    **Tip:** You can use scientific notation for large numbers (e.g., `15.7e9` for 15.7 billion)
    """)
    
    # Example data
    st.subheader("📖 Example: Apple (AAPL)")
    st.markdown("""
    | Field | Example Value |
    |-------|---------------|
    | Current Price | $150 |
    | Shares Outstanding | 15,700,000,000 |
    | Net Income | $97,000,000,000 |
    | Revenue | $391,000,000,000 |
    | Free Cash Flow | $99,800,000,000 |
    | Annual Dividend | $0.96 |
    | 5-Year Growth | 8% |
    | Terminal Growth | 2.5% |
    | Discount Rate | 9% |
    """)

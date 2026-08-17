"""
bill_generator.py
Generates a formatted PDF receipt using reportlab.
Currency: Malaysian Ringgit (RM)
Tax:      SST 6%
"""
from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

SST_RATE = 0.06


def generate_pdf(
    bill: list[dict],
    total_price: float,
    total_items: int,
    subtotal: float | None = None,
    sst: float | None = None,
) -> BytesIO:
    """
    Build a PDF receipt and return it as a BytesIO buffer.

    Each *bill* entry must have keys: Product, Quantity, Unit_Price, Total.
    If *subtotal* / *sst* are not provided they are derived from *total_price*
    by back-calculating the 6% SST so the function stays backward-compatible.
    """
    if subtotal is None:
        subtotal = total_price / (1 + SST_RATE)
    if sst is None:
        sst = subtotal * SST_RATE

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, y, "Automatic Vision-based Checkout — Receipt")
    y -= 10
    c.setLineWidth(1)
    c.line(50, y, width - 50, y)
    y -= 30

    # Column headers
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Product")
    c.drawString(270, y, "Qty")
    c.drawString(320, y, "Unit Price")
    c.drawString(430, y, "Total")
    y -= 5
    c.line(50, y, width - 50, y)
    y -= 20

    # Line items
    c.setFont("Helvetica", 11)
    for item in bill:
        c.drawString(50, y, str(item["Product"]))
        c.drawString(270, y, str(item["Quantity"]))
        c.drawString(320, y, f"RM {item['Unit_Price']:.2f}")
        c.drawString(430, y, f"RM {item['Total']:.2f}")
        y -= 20
        if y < 120:
            c.showPage()
            y = height - 50

    # Footer
    y -= 10
    c.line(50, y, width - 50, y)
    y -= 22

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Total Items: {total_items}")
    c.drawString(320, y, "Subtotal:")
    c.drawString(430, y, f"RM {subtotal:.2f}")
    y -= 18

    c.drawString(320, y, "SST (6%):")
    c.drawString(430, y, f"RM {sst:.2f}")
    y -= 6
    c.line(320, y, width - 50, y)
    y -= 18

    c.setFont("Helvetica-Bold", 13)
    c.drawString(320, y, "Grand Total:")
    c.drawString(430, y, f"RM {total_price:.2f}")

    c.save()
    buffer.seek(0)
    return buffer

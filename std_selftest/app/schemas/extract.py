from enum import Enum

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="售后描述原文")


class RequestType(str, Enum):
    REFUND = "退款"
    EXCHANGE = "换货"
    REPAIR = "维修"
    COMPLAINT = "投诉"
    OTHER = "其他"


class AfterSalesTicket(BaseModel):
    order_id: str | None = Field(description="订单号，仅当原文明确出现时提取")
    request_type: RequestType = Field(description="诉求类型")
    expected_solution: str = Field(description="一句话概括用户期望的处理方案")

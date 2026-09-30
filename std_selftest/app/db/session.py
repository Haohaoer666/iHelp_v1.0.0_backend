"""
初始化项目数据库：自动检查并创建 MySQL 数据库、通过 ORM 生成全部数据表，并在 faq 空表中插入预设问答种子数据。
"""

import aiomysql
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.db.base import Base

logger = logging.getLogger(__name__)

# 创建异步数据库引擎
engine = create_async_engine(
    settings.async_database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True,
)

# 用 SQLAlchemy 创建异步数据库会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def ensure_database() -> None:
    connection = await aiomysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA "
                "WHERE SCHEMA_NAME = %s",
                (settings.mysql_database,),
            )
            if await cursor.fetchone() is None:
                await cursor.execute(
                    f"CREATE DATABASE `{settings.mysql_database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
    finally:
        await connection.ensure_closed()


async def seed_faqs() -> None:
    from app.db.models import FAQ

    defaults = [
        ("退货政策是什么", "支持签收后 7 天内无理由退货，商品需保持完好，运费以平台售后规则为准。", "售后"),
        ("怎么申请退款", "在订单详情页点击申请售后，选择退款并提交原因，审核通过后原路退回。", "售后"),
        ("换货流程怎么走", "签收后 15 天内可申请换货，需提供商品问题照片，审核后寄回并重新发出。", "售后"),
        ("订单发货后多久能到", "现货商品通常在 48 小时内发货，运输时效根据地区不同，一般为 2 至 5 天。", "物流"),
        ("商品支持七天无理由吗", "符合平台规则且不影响二次销售的商品支持七天无理由退货。", "售后"),
    ]
    async with AsyncSessionLocal() as db:
        count = await db.scalar(select(func.count()).select_from(FAQ))
        if count:
            return
        for question, answer, category in defaults:
            db.add(FAQ(question=question, answer=answer, category=category))
        await db.commit()


async def init_db() -> None:
    from app.db import models  # noqa: F401
    from app.db.migrations import ensure_ch07_schema

    await ensure_database()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    await ensure_ch07_schema()
    await seed_faqs()


async def dispose_db() -> None:
    await engine.dispose()

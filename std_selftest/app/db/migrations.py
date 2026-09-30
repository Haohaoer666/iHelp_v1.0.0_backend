
"""
    是旧数据库升级到第七章表结构的启动兼容层，负责幂等地为 conversations 表补加 updated_at、summary_upto_msg_id 和 layer1_from_msg_id 三个字段。
"""

from sqlalchemy import inspect, text

from app.db.session import engine


CH07_CONVERSATION_COLUMNS = {
    "updated_at": "ALTER TABLE conversations ADD COLUMN updated_at DATETIME NULL",
    "summary_upto_msg_id": (
        "ALTER TABLE conversations ADD COLUMN summary_upto_msg_id INTEGER NULL"
    ),
    "layer1_from_msg_id": (
        "ALTER TABLE conversations ADD COLUMN layer1_from_msg_id INTEGER NULL"
    ),
}

#   是旧数据库升级到第七章表结构的启动兼容层，负责幂等地为 conversations 表补加 updated_at、summary_upto_msg_id 和 layer1_from_msg_id 三个字段。
async def ensure_ch07_schema() -> None:
    async with engine.begin() as connection:
        existing = await connection.run_sync(
            lambda sync_connection: {
                column["name"]
                for column in inspect(sync_connection).get_columns(
                    "conversations"
                )
            }
        )
        for column_name, statement in CH07_CONVERSATION_COLUMNS.items():
            if column_name not in existing:
                await connection.execute(text(statement))


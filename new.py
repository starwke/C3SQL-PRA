import asyncio
import hashlib
import json
import logging

from datetime import datetime, timedelta

from sqlalchemy import create_engine, select, update, func, text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


class MemoryPoller:
    """Polls memory_meta_messages and extracts memories."""

    def __init__(
        self,
        db_url: str,
        memory_service,
        poll_interval: int = 30,
        batch_size: int = 50,
        session_merge_limit: int = 1,
        history_limit: int = 5,
        processing_timeout_minutes: int = 10,
    ):
        self.engine = create_engine(
            db_url,
            pool_pre_ping=True,
        )

        self.memory_service = memory_service
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.session_merge_limit = session_merge_limit
        self.history_limit = history_limit
        self.processing_timeout_minutes = processing_timeout_minutes

        self._running = False
        self._task = None

    async def start(self) -> None:
        """Start the background polling loop."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

        log.info(
            "MemoryPoller started "
            "(interval=%ds, batch_size=%d, merge_limit=%d, "
            "history_limit=%d, processing_timeout=%dm)",
            self.poll_interval,
            self.batch_size,
            self.session_merge_limit,
            self.history_limit,
            self.processing_timeout_minutes,
        )

    async def stop(self) -> None:
        """Stop the background polling loop."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        log.info("MemoryPoller stopped")

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                count = await self._poll_once()

                if count > 0:
                    log.info(
                        "Processed %d sessions",
                        count,
                    )

            except Exception:
                log.exception("MemoryPoller polling error")

            await asyncio.sleep(self.poll_interval)

    async def _poll_once(self) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._poll_sync,
        )

    # ==================================================
    # MySQL Session 命名锁
    # ==================================================

    @staticmethod
    def _get_lock_name(session_id: str) -> str:
        """
        生成 MySQL 命名锁。

        MySQL 锁名称最大长度为 64 字符。
        """
        digest = hashlib.sha256(
            session_id.encode("utf-8")
        ).hexdigest()[:48]

        return "memory_poller_" + digest

    def _acquire_session_lock(self, conn, session_id: str) -> bool:
        """获取 Session 级 MySQL 命名锁。"""
        lock_name = self._get_lock_name(session_id)

        acquired = conn.execute(
            text("SELECT GET_LOCK(:lock_name, 0)"),
            {"lock_name": lock_name},
        ).scalar()

        return acquired == 1

    def _release_session_lock(self, conn, session_id: str) -> None:
        """释放 Session 级 MySQL 命名锁。"""
        lock_name = self._get_lock_name(session_id)

        result = conn.execute(
            text("SELECT RELEASE_LOCK(:lock_name)"),
            {"lock_name": lock_name},
        ).scalar()

        if result != 1:
            log.warning(
                "Session lock release result=%s, session_id=%s",
                result,
                session_id,
            )

    # ==================================================
    # 主轮询流程
    # ==================================================

    def _poll_sync(self) -> int:
        """
        同步轮询。

        1. 回收超时 processing 记录。
        2. 查询最近活跃的 Session。
        3. 逐个处理 Session。
        """
        self._reclaim_stale_processing()

        with Session(self.engine) as db:
            pending_rows = db.execute(
                select(
                    MemoryMetaMessage.session_id,
                    func.max(
                        MemoryMetaMessage.created_at
                    ).label("latest_created_at"),
                )
                .where(
                    MemoryMetaMessage.status == "pending",
                    ~MemoryMetaMessage.space_id.like("app\\_%"),
                )
                .group_by(
                    MemoryMetaMessage.session_id
                )
                .order_by(
                    func.max(
                        MemoryMetaMessage.created_at
                    ).desc()
                )
                .limit(self.batch_size)
            ).all()

            session_ids = [
                row.session_id
                for row in pending_rows
            ]

        if not session_ids:
            return 0

        processed = 0

        for session_id in session_ids:
            try:
                if self._process_session(session_id):
                    processed += 1

            except Exception:
                log.exception(
                    "Failed to process session %s",
                    session_id,
                )

        return processed

    # ==================================================
    # 超时 processing 回收
    # ==================================================

    def _reclaim_stale_processing(self) -> None:
        """
        回收异常退出后遗留的 processing 记录。

        不能直接批量将所有超时记录改成 pending。
        必须先获取对应 Session 的命名锁：

        - 获取成功：说明没有其他实例持有该 Session 的处理锁，可以回收。
        - 获取失败：说明仍有实例正在处理，不回收。
        """
        cutoff = datetime.now() - timedelta(
            minutes=self.processing_timeout_minutes
        )

        with Session(self.engine) as db:
            stale_sessions = db.execute(
                select(
                    MemoryMetaMessage.session_id
                )
                .where(
                    MemoryMetaMessage.status == "processing",
                    MemoryMetaMessage.extracted_at.is_not(None),
                    MemoryMetaMessage.extracted_at < cutoff,
                )
                .distinct()
            ).scalars().all()

        reclaimed_total = 0

        for session_id in stale_sessions:
            if not session_id:
                continue

            try:
                # AUTOCOMMIT 保证命名锁连接独立于 ORM 事务。
                with self.engine.connect().execution_options(
                    isolation_level="AUTOCOMMIT"
                ) as conn:

                    acquired = self._acquire_session_lock(
                        conn,
                        session_id,
                    )

                    if not acquired:
                        continue

                    try:
                        # 获取锁后再次检查时间和状态，防止使用过期查询结果。
                        with Session(self.engine) as db:
                            result = db.execute(
                                update(MemoryMetaMessage)
                                .where(
                                    MemoryMetaMessage.session_id == session_id,
                                    MemoryMetaMessage.status == "processing",
                                    MemoryMetaMessage.extracted_at.is_not(None),
                                    MemoryMetaMessage.extracted_at < cutoff,
                                )
                                .values(
                                    status="pending",
                                    extracted_at=None,
                                )
                            )

                            db.commit()

                            if result.rowcount:
                                reclaimed_total += result.rowcount

                                log.warning(
                                    "Reclaimed stale processing records: "
                                    "session=%s count=%d",
                                    session_id,
                                    result.rowcount,
                                )

                    finally:
                        self._release_session_lock(
                            conn,
                            session_id,
                        )

            except Exception:
                log.exception(
                    "Failed to reclaim session %s",
                    session_id,
                )

        if reclaimed_total:
            log.warning(
                "Total reclaimed stale records: %d",
                reclaimed_total,
            )

    # ==================================================
    # Session 处理入口
    # ==================================================

    def _process_session(self, session_id: str) -> bool:
        """
        获取 Session 级命名锁，然后处理该 Session。

        命名锁在整个提取期间保持持有，
        但数据库事务不会一直保持打开。
        """
        try:
            with self.engine.connect().execution_options(
                isolation_level="AUTOCOMMIT"
            ) as lock_conn:

                acquired = self._acquire_session_lock(
                    lock_conn,
                    session_id,
                )

                if not acquired:
                    log.debug(
                        "Session is being processed by another instance: %s",
                        session_id,
                    )
                    return False

                try:
                    return self._process_locked_session(
                        session_id
                    )

                finally:
                    self._release_session_lock(
                        lock_conn,
                        session_id,
                    )

        except Exception:
            log.exception(
                "Session processing failed: %s",
                session_id,
            )
            raise

    def _process_locked_session(self, session_id: str) -> bool:
        """在持有 Session 命名锁的情况下执行提取。"""

        # 第一阶段：短事务领取消息
        records = self._claim_records(session_id)

        if not records:
            return False

        ids = [
            record["id"]
            for record in records
        ]

        try:
            # 第二阶段：事务外执行 LLM、Embedding、ES
            self._extract_session_memory(
                session_id,
                records,
            )

        except Exception:
            log.exception(
                "Memory extraction failed: session=%s",
                session_id,
            )

            # 提取失败，恢复 pending
            self._mark_pending(ids)

            return False

        # 第三阶段：独立短事务更新 done
        # 注意：如果这里失败，不能再把消息直接恢复 pending。
        # 保留 processing，由超时回收机制处理。
        self._mark_done(ids)

        return True

    # ==================================================
    # 领取消息：pending -> processing
    # ==================================================

    def _claim_records(self, session_id: str):
        """
        领取当前 Session 最新的 pending 消息。

        extracted_at 在领取时记录开始处理时间。
        """
        with Session(self.engine) as db:
            try:
                ids = db.execute(
                    select(
                        MemoryMetaMessage.id
                    )
                    .where(
                        MemoryMetaMessage.session_id == session_id,
                        MemoryMetaMessage.status == "pending",
                    )
                    .order_by(
                        MemoryMetaMessage.chat_seq.is_(None),
                        MemoryMetaMessage.chat_seq.desc(),
                        MemoryMetaMessage.created_at.desc(),
                    )
                    .limit(self.session_merge_limit)
                ).scalars().all()

                if not ids:
                    db.commit()
                    return []

                now = datetime.now()

                result = db.execute(
                    update(MemoryMetaMessage)
                    .where(
                        MemoryMetaMessage.id.in_(ids),
                        MemoryMetaMessage.status == "pending",
                    )
                    .values(
                        status="processing",
                        extracted_at=now,
                    )
                )

                # 领取数量不一致时回滚，避免处理不完整的数据。
                if result.rowcount != len(ids):
                    db.rollback()

                    log.warning(
                        "Claim count mismatch: session=%s expected=%d actual=%d",
                        session_id,
                        len(ids),
                        result.rowcount,
                    )

                    return []

                rows = db.execute(
                    select(MemoryMetaMessage)
                    .where(
                        MemoryMetaMessage.id.in_(ids),
                    )
                ).scalars().all()

                # 恢复原始排序，避免 IN 查询打乱顺序。
                record_map = {
                    row.id: row
                    for row in rows
                }

                records = []

                for record_id in ids:
                    row = record_map.get(record_id)

                    if row is None:
                        continue

                    records.append(
                        {
                            "id": row.id,
                            "message": row.message,
                            "chat_id": row.chat_id,
                            "chat_seq": row.chat_seq,
                            "space_id": row.space_id,
                            "agent_id": row.agent_id,
                        }
                    )

                db.commit()

                return records

            except Exception:
                db.rollback()
                raise

    # ==================================================
    # 实际记忆提取：不持有数据库事务
    # ==================================================

    def _extract_session_memory(
        self,
        session_id: str,
        records: list,
    ) -> None:
        """
        解析消息、获取历史上下文、调用 memory.add()。

        所有数据库查询使用独立短 Session。
        memory.add() 在数据库事务之外执行。
        """
        all_messages = []
        chat_ids = []

        for record in records:
            msgs = (
                json.loads(record["message"])
                if isinstance(record["message"], str)
                else record["message"]
            )

            if isinstance(msgs, list):
                for message in msgs:
                    if (
                        message.get("content")
                        and len(message["content"]) > MAX_MSG_CHARS
                    ):
                        message["content"] = truncate_head(
                            message["content"]
                        )

                    all_messages.append(message)

            chat_ids.append(record["chat_id"])

        if not all_messages:
            raise ValueError(
                f"Session {session_id} has no valid messages"
            )

        # 当前批次第一条是 chat_seq 最大的记录，即最新记录。
        latest = records[0]

        # 查询当前批次之前最近的历史 chat_id。
        history_chat_ids = []

        if latest["chat_seq"] is not None:
            with Session(self.engine) as db:
                history_chat_ids = list(
                    db.execute(
                        select(
                            MemoryMetaMessage.chat_id
                        )
                        .where(
                            MemoryMetaMessage.session_id == session_id,
                            MemoryMetaMessage.chat_seq < latest["chat_seq"],
                        )
                        .order_by(
                            MemoryMetaMessage.chat_seq.desc()
                        )
                        .limit(self.history_limit)
                    ).scalars().all()
                )

        metadata = {
            "source": "auto",
            "chat_ids": history_chat_ids + chat_ids,
            "created_at": datetime.now().isoformat(),
            "session_id": session_id,
        }

        log.info(
            "开始提取记忆: session_id=%s chat_ids=%s "
            "space_id=%s agent_id=%s records=%d msgs=%d",
            session_id,
            metadata["chat_ids"],
            latest["space_id"],
            latest["agent_id"],
            len(records),
            len(all_messages),
        )

        # 保留原有 ExternalHistoryDB 上下文注入逻辑。
        if latest["chat_seq"] is not None:
            self.memory_service.memory.db.set_current_batch(
                session_id,
                latest["chat_seq"],
            )

        # 重要：此处不在任何 SQLAlchemy Session 事务中。
        add_result = self.memory_service.memory.add(
            all_messages,
            user_id=latest["space_id"],
            agent_id=latest["agent_id"] or None,
            metadata=metadata,
        )

        extracted_count = (
            len(add_result.get("results", []))
            if isinstance(add_result, dict)
            else 0
        )

        log.info(
            "提取完成并写入ES: session_id=%s records=%d "
            "提炼记忆条数=%d",
            session_id,
            len(records),
            extracted_count,
        )

    # ==================================================
    # 成功回写：processing -> done
    # ==================================================

    def _mark_done(self, ids: list) -> None:
        """提取成功后，记录实际完成时间。"""

        with Session(self.engine) as db:
            try:
                result = db.execute(
                    update(MemoryMetaMessage)
                    .where(
                        MemoryMetaMessage.id.in_(ids),
                        MemoryMetaMessage.status == "processing",
                    )
                    .values(
                        status="done",
                        extracted_at=datetime.now(),
                    )
                )

                db.commit()

                if result.rowcount != len(ids):
                    log.warning(
                        "Done状态回写数量不一致: expected=%d actual=%d",
                        len(ids),
                        result.rowcount,
                    )

            except Exception:
                db.rollback()
                raise

    # ==================================================
    # 失败回写：processing -> pending
    # ==================================================

    def _mark_pending(self, ids: list) -> None:
        """提取异常时恢复待处理状态。"""

        with Session(self.engine) as db:
            try:
                db.execute(
                    update(MemoryMetaMessage)
                    .where(
                        MemoryMetaMessage.id.in_(ids),
                        MemoryMetaMessage.status == "processing",
                    )
                    .values(
                        status="pending",
                        extracted_at=None,
                    )
                )

                db.commit()

            except Exception:
                db.rollback()

                log.exception(
                    "Failed to restore pending records: ids=%s",
                    ids,
                )
                raise
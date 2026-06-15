"""Excel 行级分块单测。"""

import tempfile
import unittest
from pathlib import Path


class TestExcelRowLevelChunking(unittest.TestCase):
    """验证 XlsxReader + IngestionPipeline 对 Excel 的行级分块行为。"""

    @classmethod
    def setUpClass(cls):
        from src.core.config import load_config
        load_config("development")

    @classmethod
    def tearDownClass(cls):
        # 重置 lru_cache 单例，防止 config 变更后跨测试类残留旧实例
        try:
            from src.core.container import clear_all_caches
            clear_all_caches()
        except Exception:
            pass

    def _make_xlsx(self, rows: list[list]) -> Path:
        """在临时目录构造一个单 Sheet xlsx，返回文件路径。"""
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        for row in rows:
            ws.append(row)
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        wb.save(tmp.name)
        tmp.close()
        return Path(tmp.name)

    def tearDown(self):
        # 清理临时文件
        if hasattr(self, "_tmp_path") and self._tmp_path.exists():
            self._tmp_path.unlink(missing_ok=True)

    def test_chunk_count_equals_data_rows(self):
        """chunk 数量应等于非空数据行数，不含表头。"""
        from src.ingestion.pipeline import IngestionPipeline
        path = self._make_xlsx([
            ["标题", "模块", "步骤", "预期"],
            ["用例A", "登录", "点击登录", "跳转首页"],
            ["用例B", "搜索", "输入关键词", "显示结果"],
            ["用例C", "退出", "点击退出", "回到登录页"],
        ])
        self._tmp_path = path
        pipeline = IngestionPipeline()
        _, chunks = pipeline.ingest(path)
        self.assertEqual(len(chunks), 3)

    def test_each_chunk_contains_only_its_own_row(self):
        """每个 chunk 只包含自己那行的标题，不包含相邻行。"""
        from src.ingestion.pipeline import IngestionPipeline
        path = self._make_xlsx([
            ["标题", "模块"],
            ["追番Card页面逻辑", "追番"],
            ["2x2模块展示", "金刚区"],
            ["顶部bar交互", "顶部导航"],
        ])
        self._tmp_path = path
        pipeline = IngestionPipeline()
        _, chunks = pipeline.ingest(path)

        self.assertIn("追番Card页面逻辑", chunks[0].content)
        self.assertNotIn("2x2模块", chunks[0].content)
        self.assertNotIn("顶部bar", chunks[0].content)

        self.assertIn("2x2模块展示", chunks[1].content)
        self.assertNotIn("追番Card", chunks[1].content)

        self.assertIn("顶部bar交互", chunks[2].content)
        self.assertNotIn("2x2模块", chunks[2].content)

    def test_chunk_metadata_contains_required_fields(self):
        """每个 chunk 的 metadata 须包含 sheet_name、row_index、row_id。"""
        from src.ingestion.pipeline import IngestionPipeline
        path = self._make_xlsx([
            ["用例名", "优先级"],
            ["登录成功", "P0"],
            ["登录失败", "P1"],
        ])
        self._tmp_path = path
        pipeline = IngestionPipeline()
        _, chunks = pipeline.ingest(path)

        for chunk in chunks:
            self.assertIn("sheet_name", chunk.metadata)
            self.assertIn("row_index", chunk.metadata)
            self.assertIn("row_id", chunk.metadata)
            self.assertIn("source_format", chunk.metadata)
            self.assertEqual(chunk.metadata["source_format"], "xlsx")

    def test_empty_rows_are_skipped(self):
        """全空行不生成 chunk。"""
        from src.ingestion.pipeline import IngestionPipeline
        path = self._make_xlsx([
            ["标题", "模块"],
            ["用例A", "登录"],
            [None, None],       # 全空行
            ["用例B", "搜索"],
        ])
        self._tmp_path = path
        pipeline = IngestionPipeline()
        _, chunks = pipeline.ingest(path)
        self.assertEqual(len(chunks), 2)

    def test_non_excel_still_uses_chunker(self):
        """txt 文件仍走 ChineseChunker，不走 segments 路径。"""
        from src.ingestion.readers.base import Document
        from src.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write("这是第一句话。这是第二句话。这是第三句话。")
            txt_path = Path(f.name)
        try:
            doc, chunks = pipeline.ingest(txt_path)
            self.assertEqual(doc.segments, [])
            self.assertGreater(len(chunks), 0)
        finally:
            txt_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

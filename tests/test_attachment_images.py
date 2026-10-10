"""Multi-attachment validation for text-input images.

Covers sanitize_images (per-file validation, limits, error reporting) and
its handoff to create_batch_input. Attachments are in-memory base64
throughout: no test may write outside its own tmp dir, and production
chat_history must never be touched (no tests here import the history
store at all).
"""

import base64
import os
import unittest

from src.open_llm_vtuber.conversations.conversation_utils import (
    MAX_IMAGE_FILE_BYTES,
    MAX_IMAGES_PER_MESSAGE,
    MAX_IMAGES_TOTAL_BYTES,
    create_batch_input,
    sanitize_images,
)


def data_url(n_bytes, mime="image/png"):
    raw = base64.b64encode(bytes(n_bytes)).decode("ascii")
    return f"data:{mime};base64,{raw}"


def img(name="a.png", n=100, mime="image/png", source="upload", **extra):
    entry = {
        "source": source,
        "data": data_url(n, mime),
        "mime_type": mime,
        "name": name,
        "size": n,
    }
    entry.update(extra)
    return entry


class SanitizeHappyPathTests(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertEqual(sanitize_images(None), ([], []))
        self.assertEqual(sanitize_images([]), ([], []))

    def test_multiple_valid_files_keep_order_and_identity(self):
        images = [img("b.png"), img("a.png"), img("c.png")]
        valid, errors = sanitize_images(images, request_id="req-1")
        self.assertEqual(errors, [])
        self.assertEqual(len(valid), 3)
        # Untouched originals, order preserved.
        self.assertIs(valid[0], images[0])
        self.assertIs(valid[2], images[2])
        self.assertEqual([v["name"] for v in valid], ["b.png", "a.png", "c.png"])

    def test_same_name_and_same_content_never_overwrite(self):
        blob = data_url(50)
        images = [
            {
                "source": "upload",
                "data": blob,
                "mime_type": "image/png",
                "name": "foto.png",
            },
            {
                "source": "upload",
                "data": blob,
                "mime_type": "image/png",
                "name": "foto.png",
            },
        ]
        valid, errors = sanitize_images(images)
        self.assertEqual(errors, [])
        self.assertEqual(len(valid), 2)

    def test_camera_and_screen_sources_accepted(self):
        images = [
            img("cam.jpg", source="camera", mime="image/jpeg"),
            img("scr.jpg", source="screen", mime="image/jpeg"),
        ]
        valid, errors = sanitize_images(images)
        self.assertEqual(errors, [])
        self.assertEqual(len(valid), 2)


class SanitizeInvalidTests(unittest.TestCase):
    def test_not_a_list(self):
        valid, errors = sanitize_images({"source": "upload"}, request_id="r")
        self.assertEqual(valid, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["reason"], "not-a-list")

    def test_mixed_valid_and_invalid(self):
        images = [
            img("good.png", n=100),
            {
                "source": "upload",
                "data": data_url(10),
                "mime_type": "application/pdf",
                "name": "doc.pdf",
            },
            {
                "source": "upload",
                "data": "not-a-data-url",
                "mime_type": "image/png",
                "name": "broken.png",
            },
            {
                "source": "telepathy",
                "data": data_url(10),
                "mime_type": "image/png",
                "name": "weird.png",
            },
            "just-a-string",
            {"source": "upload", "mime_type": "image/png", "name": "nodata.png"},
        ]
        valid, errors = sanitize_images(images, request_id="req-9")
        self.assertEqual([v["name"] for v in valid], ["good.png"])
        by_name = {e["name"]: e["reason"] for e in errors}
        self.assertEqual(
            by_name,
            {
                "doc.pdf": "unsupported-type",
                "broken.png": "bad-data",
                "weird.png": "bad-source",
                "image #5": "not-a-dict",
                "nodata.png": "bad-data",
            },
        )
        self.assertEqual(len(errors), 5)

    def test_malformed_never_crashes(self):
        for bad in (
            ["x"],
            [None],
            [123],
            [{"source": "upload"}],
            [{"source": "upload", "data": None, "mime_type": None}],
            [
                {
                    "source": "upload",
                    "data": "data:image/png;base64,!!!",
                    "mime_type": "image/png",
                }
            ],
        ):
            valid, errors = sanitize_images(bad)
            self.assertEqual(valid, [])
            self.assertEqual(len(errors), 1)


class SanitizeLimitTests(unittest.TestCase):
    def test_per_file_limit(self):
        big = img("big.png", n=MAX_IMAGE_FILE_BYTES + 1)
        ok = img("ok.png", n=100)
        valid, errors = sanitize_images([big, ok])
        self.assertEqual([v["name"] for v in valid], ["ok.png"])
        self.assertEqual(
            [(e["name"], e["reason"]) for e in errors], [("big.png", "too-large")]
        )

    def test_boundary_ten_accepted_eleventh_refused(self):
        images = [img(f"f{i}.png", n=10) for i in range(10)]
        valid, errors = sanitize_images(images)
        self.assertEqual(len(valid), 10)
        self.assertEqual(errors, [])
        images.append(img("f10.png", n=10))
        valid, errors = sanitize_images(images)
        self.assertEqual(len(valid), 10)
        self.assertEqual([(e["name"], e["reason"]) for e in errors],
                         [("f10.png", "too-many")])

    def test_small_counts_always_pass(self):
        for n in (1, 2, 5):
            valid, errors = sanitize_images(
                [img(f"f{i}.png", n=10) for i in range(n)])
            self.assertEqual(len(valid), n, f"n={n}")
            self.assertEqual(errors, [], f"n={n}")

    def test_count_cap_keeps_first_in_order(self):
        images = [img(f"f{i}.png", n=10) for i in range(MAX_IMAGES_PER_MESSAGE + 2)]
        valid, errors = sanitize_images(images)
        self.assertEqual(len(valid), MAX_IMAGES_PER_MESSAGE)
        self.assertEqual(
            [v["name"] for v in valid],
            [f"f{i}.png" for i in range(MAX_IMAGES_PER_MESSAGE)],
        )
        self.assertEqual(
            [(e["name"], e["reason"]) for e in errors],
            [
                (f"f{MAX_IMAGES_PER_MESSAGE}.png", "too-many"),
                (f"f{MAX_IMAGES_PER_MESSAGE + 1}.png", "too-many"),
            ],
        )

    def test_aggregate_cap_fills_in_order(self):
        third = MAX_IMAGES_TOTAL_BYTES - 2 * (MAX_IMAGES_TOTAL_BYTES // 3)
        sizes = [MAX_IMAGES_TOTAL_BYTES // 3] * 2 + [third + 1]
        images = [img(f"f{i}.png", n=n) for i, n in enumerate(sizes)]
        valid, errors = sanitize_images(images)
        self.assertEqual([v["name"] for v in valid], ["f0.png", "f1.png"])
        self.assertEqual(
            [(e["name"], e["reason"]) for e in errors], [("f2.png", "total-too-large")]
        )

    def test_limits_are_sane_for_transport(self):
        # Wire size (base64 x4/3) must fit uvicorn's 16 MiB ws cap.
        self.assertLessEqual(MAX_IMAGES_TOTAL_BYTES * 4 // 3, 15 * 1024 * 1024)
        self.assertLessEqual(MAX_IMAGE_FILE_BYTES, MAX_IMAGES_TOTAL_BYTES)
        self.assertGreaterEqual(MAX_IMAGES_PER_MESSAGE, 2)


class SanitizeAssociationTests(unittest.TestCase):
    def test_request_id_accepted_and_valid_untouched(self):
        images = [img("a.png")]
        valid, errors = sanitize_images(images, request_id="req-abc")
        self.assertEqual(errors, [])
        self.assertIs(valid[0], images[0])

    def test_no_cross_message_leakage(self):
        first, _ = sanitize_images([img("a.png")])
        second, _ = sanitize_images([img("b.png")])
        self.assertEqual([v["name"] for v in first], ["a.png"])
        self.assertEqual([v["name"] for v in second], ["b.png"])
        self.assertIsNot(first, second)

    def test_no_filesystem_writes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before = set(os.listdir(tmp))
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                sanitize_images([img("a.png", n=1000), img("b.png", n=2000)])
                create_batch_input(
                    "hi", sanitize_images([img("c.png")])[0], from_name="User"
                )
            finally:
                os.chdir(cwd)
            self.assertEqual(set(os.listdir(tmp)), before)

    def test_handoff_to_batch_input_preserves_order_and_fields(self):
        images = [
            img("b.png", n=50, mime="image/jpeg"),
            img("a.png", n=60, mime="image/png"),
        ]
        valid, errors = sanitize_images(images)
        self.assertEqual(errors, [])
        batch = create_batch_input("lihat ini", valid, from_name="User")
        self.assertIsNotNone(batch.images)
        self.assertEqual(len(batch.images), 2)
        self.assertEqual(batch.images[0].mime_type, "image/jpeg")
        self.assertEqual(batch.images[1].mime_type, "image/png")
        self.assertTrue(batch.images[0].data.startswith("data:image/"))


if __name__ == "__main__":
    unittest.main()

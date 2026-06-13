from flask import Flask, request, jsonify
from matrix_ops import MatrixOperations

app = Flask(__name__)


def _parse_request():
    data = request.get_json(silent=True)
    if not data:
        return None, None, "请求体必须为有效的 JSON"
    matrix_a = data.get("matrix_a")
    matrix_b = data.get("matrix_b")
    if matrix_a is None or matrix_b is None:
        return None, None, "请求体必须包含 matrix_a 和 matrix_b 字段"
    return matrix_a, matrix_b, None


@app.route("/api/matrix/add", methods=["POST"])
def matrix_add():
    matrix_a, matrix_b, error = _parse_request()
    if error:
        return jsonify({"success": False, "error": error}), 400
    try:
        result = MatrixOperations.add(matrix_a, matrix_b)
        return jsonify({
            "success": True,
            "operation": "add",
            "shape_a": MatrixOperations.get_shape(matrix_a),
            "shape_b": MatrixOperations.get_shape(matrix_b),
            "result": result,
        })
    except (TypeError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"服务器内部错误: {str(e)}"}), 500


@app.route("/api/matrix/subtract", methods=["POST"])
def matrix_subtract():
    matrix_a, matrix_b, error = _parse_request()
    if error:
        return jsonify({"success": False, "error": error}), 400
    try:
        result = MatrixOperations.subtract(matrix_a, matrix_b)
        return jsonify({
            "success": True,
            "operation": "subtract",
            "shape_a": MatrixOperations.get_shape(matrix_a),
            "shape_b": MatrixOperations.get_shape(matrix_b),
            "result": result,
        })
    except (TypeError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"服务器内部错误: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "matrix-ops"})


if __name__ == "__main__":
    print("矩阵加减法服务启动中...")
    print("  POST /api/matrix/add      - 矩阵加法")
    print("  POST /api/matrix/subtract - 矩阵减法")
    print("  GET  /api/health          - 健康检查")
    app.run(host="0.0.0.0", port=5000, debug=True)

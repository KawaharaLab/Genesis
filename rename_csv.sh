#!/bin/bash

# 対象のルートディレクトリを引数から取得。指定がなければ 'data/csv' を使う。
TARGET_DIR="${1:-data/csv}"

# --execute フラグがあるかどうかをチェック
EXECUTE=false
if [ "$2" == "--execute" ]; then
    EXECUTE=true
fi

echo "Searching in '$TARGET_DIR'..."
if [ "$EXECUTE" = false ]; then
    echo "--- This is a DRY RUN. No files will be renamed. ---"
else
    echo "--- Executing rename operation. ---"
fi

# findコマンドで見つけたファイルを一つずつ処理する
find "$TARGET_DIR" -type f -name "*_*.5.csv" | while read -r filepath; do
    # ディレクトリパスとファイル名を取得
    dir=$(dirname "$filepath")
    filename=$(basename "$filepath")
    
    # 新しいファイル名を生成 (bashのパラメータ置換を利用)
    # 1. _0.5.csv -> _050.csv
    # 2. _1.5.csv -> _150.csv
    # 3. _2.5.csv -> _250.csv
    new_filename=${filename/_0.5.csv/_050.csv}
    new_filename=${new_filename/_1.5.csv/_150.csv}
    new_filename=${new_filename/_2.5.csv/_250.csv}
    
    # ファイル名が変更された場合のみ処理
    if [ "$filename" != "$new_filename" ]; then
        new_filepath="$dir/$new_filename"
        echo "Found: $filepath"
        echo "  -> Rename to: $new_filepath"
        
        if [ "$EXECUTE" = true ]; then
            # mvコマンドでファイル名を変更
            mv "$filepath" "$new_filepath"
            echo "  -> Renamed successfully."
        fi
    fi
done

echo "--------------------"
echo "Finished."
if [ "$EXECUTE" = false ]; then
    echo -e "\nTo actually rename the files, run this script with the --execute flag."
    echo "Example: ./rename.sh $TARGET_DIR --execute"
fi
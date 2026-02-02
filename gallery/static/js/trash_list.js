function restorePhoto(photoId, csrfToken) {
    if (!confirm('사진을 복구하시겠습니까?')) return;
    fetch(`/gallery/trash/${photoId}/restore/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken }
    })
    .then(res => res.json())
    .then(data => { if(data.success) location.reload(); });
}

function permanentDelete(photoId, csrfToken) {
    if (!confirm('영구 삭제하면 되돌릴 수 없습니다. 삭제하시겠습니까?')) return;
    fetch(`/gallery/trash/${photoId}/`, {
        method: 'DELETE',
        headers: { 'X-CSRFToken': csrfToken }
    })
    .then(res => res.json())
    .then(data => { if(data.success) location.reload(); });
}
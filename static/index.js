var fileInput = document.getElementById('test-image-file');
var preview = document.getElementById('test-image-preview');
preview.flag = '0';

var image = '';


// 监听change事件:
fileInput.addEventListener('change', function () {
    // 检查文件是否选择:
    if (!fileInput.value) {
        return;
    }
    // 获取File引用:
    var file = fileInput.files[0];
    // 获取File信息:
    if (file.type !== 'image/jpeg' && file.type !== 'image/png' && file.type !== 'image/gif') {
        alert('Invalid image file.');
        return;
    }

    preview.flag = '1';

    // 读取文件:
    var reader = new FileReader();
    reader.onload = function(e) {
        preview.innerHTML = '<img src="' + e.target.result + '" height=300 />\n';
        image = e.target.result;
    };
    // 以DataURL的形式读取文件:
    reader.readAsDataURL(file);
    
    function img_to_backend(){
        var formData = new FormData();
        formData.append("img", fileInput.files[0]);
        alert("ok");
        
        $.ajax({
            url: '/initialize',
            type: 'post',
            data: formData,
            processData: false,
            contentType: false,
            success: function (msg) {
                alert(msg);
            }
            
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
        alert("error");
        });
    }
});

function Upload_params(){
    var param1 = document.getElementById("param1").value;
    var param2 = document.getElementById("param2").value;
    var param3 = document.getElementById("param3").value;
    var param4 = document.getElementById("param4").value;
    
    if ((param1 < 0) || (param1 > 1)){
        alert("Whightening degree should be in range [0, 1]");
        return;
    }

    if ((param2 < 0) || (param2 > 1)){
        alert("lip brightening degree should be in range [0, 1]");
        return;
    }

    if ((param3 < 0) || (param3 > 1)){
        alert("face thining degree should be in range [0, 1]");
        return;
    }

    if ((param4 < 0) || (param4 > 1)){
        alert("smoothing degree should be in range [0, 1]");
        return;
    }

    if (preview.flag == '0'){
        alert("Please provide your image first.");
        return;
    }


    function send_to_backend(param1, param2, param3, param4){
        var formdata = new FormData();
        var sharpen_val = 0.35
        formdata.append("img", fileInput.files[0]);
        formdata.append("whighten", param1);
        formdata.append("lip_brighten", param2);
        formdata.append("thin", param3);
        formdata.append("smooth", param4);
        formdata.append("sharpen", sharpen_val);

        alert("ok");

        $.ajax({
            url: '/process',
            type: 'post',
            data: formdata,
            processData: false,
            contentType: false,
            
        }).done(function (data){
            var image_src = "data:image/png;base64," + data;
            preview.innerHTML += "<img src=" + image_src + "</img>\n";
        }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
            alert("error");
        });
    }

    send_to_backend(param1, param2, param3, param4);
}
$(document).ready(function() {
  $('#result').hide()
  $('#form-upload').on('submit', function(evt) {
    evt.preventDefault();
    var formData = new FormData($('form')[0]);
    $.ajax({
      xhr: function() {
        var xhr = new window.XMLHttpRequest();
        xhr.upload.addEventListener("progress", function(e) {
          if (e.lengthComputable) {
            console.log('Bytes Loaded        : ' + e.loaded);
            console.log('Total Size          : ' + e.total);
            console.log('Percentage Uploaded : ' + e.loaded / e.total);

            var percent = Math.round((e.loaded / e.total) * 100);
            $('#progressBar').attr('aria-valuenow', percent).css('width', percent + '%').text(percent + '%');
            if (percent === 100) {
              $("#loading_training").show();
//              $("#progressBar").hide();
            }
          }
        });
        return xhr;
      },
      type:'POST',
      url:'/',
      data: formData,
      processData:false,
      contentType:false,
      success:function(data) {
        alert('Training Completed!');
        $("#loading_training").hide();
        $('#progressBar').attr('aria-valuenow', '').css('width', '0%').text('0%');
        $('#result').text(data.train);
        $('#result').show();
      }
    });
  });

  $('#form-predict').on('submit', function(evt) {
    evt.preventDefault();
    var formData = new FormData($('form')[0]);
    $.ajax({
      xhr: function() {
        var xhr = new window.XMLHttpRequest();
        xhr.upload.addEventListener("progress", function(e) {
          if (e.lengthComputable) {

            var percent = Math.round((e.loaded / e.total) * 100);
            $('#progressBar').attr('aria-valuenow', percent).css('width', percent + '%').text(percent + '%');
            if (percent === 100) {
              $("#loading_predict").show();
//              $("#progressBar").hide();
            }
          }
        });
        return xhr;
      },
      type:'POST',
      url:'/predict',
      data: formData,
      processData:false,
      contentType:false,
      success:function(data) {
        alert('Predict Completed!');
        $("#loading_predict").hide();
        $('#progressBar').attr('aria-valuenow', '').css('width', '0%').text('0%');
        $('#result').show();
        $('#acc').text(data.accuracy);
        $('#detected').text(data.class)
      }
    });
  });
});
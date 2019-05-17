$(document).ready(function() {
  alert('abc');
  $('#form-upload').on('submit', function(evt) {
    evt.preventDefault();
    var formData = new FormData($('form')[0]);
    $.ajax({
      xhr: function() {
        var xhr = new window.XMLHttpRequest();
        return xhr;
      },
      type:'POST',
      url:'/',
      data: formData,
      processData:false,
      contentType:false,
      success:function(data) {
        alert('Completed!');
        $('#result').text(data.train);
      }
    });
  });
});